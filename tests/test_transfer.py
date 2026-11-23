"""Board-size transfer and accepted-position stage boundaries."""

import dataclasses
import json

import numpy as np
import pytest
import torch
from test_core import small

from zero_systems.config import Evaluation, System
from zero_systems.env import task_spec
from zero_systems.evaluate import evaluate, fingerprint, load_candidate
from zero_systems.learn import Learner
from zero_systems.live import watch_once
from zero_systems.model import build_model, initialize
from zero_systems.run import TrainingRun, train
from zero_systems.storage import load_checkpoint


@pytest.mark.parametrize("network", ["residual", "transformer"])
def test_spatial_transfer_copies_every_selected_tensor(network):
    torch.set_num_threads(1)
    config = small(network=network, prediction_head="spatial", board_size=5)
    source = build_model(config, task_spec(config))
    source.train()
    source.initial_inference(torch.randn(2, 6, 5, 5), torch.empty(2, 0))
    for side in (5, 7, 9):
        target_config = dataclasses.replace(config, board_size=side)
        for components in (
            ("representation",),
            ("representation", "dynamics"),
            ("representation", "dynamics", "prediction"),
        ):
            target = build_model(target_config, task_spec(target_config))
            fresh = {k: v.clone() for k, v in target.state_dict().items()}
            assert {k: v.shape for k, v in fresh.items()} == {
                k: v.shape for k, v in source.state_dict().items()
            }
            report = initialize(
                target,
                source.state_dict(),
                components,
                source_task=source.task,
                source_config=config,
            )
            for key, value in target.state_dict().items():
                selected = key.split(".")[0] in components
                assert torch.equal(value, source.state_dict()[key] if selected else fresh[key])
                assert key in report["copied" if selected else "reset"]
            assert not report["expanded"]
            state, policy, value = target.initial_inference(
                torch.randn(2, 6, side, side), torch.empty(2, 0)
            )
            next_state, next_policy, next_value, reward = target.recurrent_inference(
                state, torch.tensor([0, side * side])
            )
            assert policy.shape == next_policy.shape == (2, side * side + 1)
            assert state.shape == next_state.shape == (2, config.channels, side, side)
            loss = (
                policy.square().mean()
                + value.square().mean()
                + next_policy.square().mean()
                + next_value.square().mean()
            )
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(reward).all()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in target.parameters()
            )
    weights = dict(source.state_dict())
    weights.pop(next(iter(weights)))
    with pytest.raises(ValueError, match="tensors"):
        initialize(source, weights, source_task=source.task, source_config=config)
    for field, value in (
        ("history", 3),
        ("condition_komi", True),
        ("value_perspective", "to_play"),
        ("discount", 0.9),
        ("network", "mlp"),
    ):
        with pytest.raises(ValueError, match="architecture"):
            initialize(
                source,
                source.state_dict(),
                source_task=source.task,
                source_config=dataclasses.replace(config, **{field: value}),
            )


@pytest.mark.parametrize("network", ["residual", "transformer"])
def test_position_budget_transfer_resume_and_final_evaluation(tmp_path, monkeypatch, network):
    source_config = small(network=network, prediction_head="spatial", updates=2)
    source_dir = tmp_path / "source"
    train(source_config, System(actor_batch=4), source_dir)
    source_path = source_dir / "models/step_00000002.pt"
    config = dataclasses.replace(
        source_config,
        board_size=5,
        schedule="async",
        updates=1000,
        initialize=str(source_path),
        samples_per_position=0.3,
        warmup_positions=4,
        position_lead=14,
        max_positions=60,
        replay_games=1,
        publish_updates=2,
    )
    system = System(
        actor_devices=("cpu",),
        actor_batch=2,
        microbatch=2,
        checkpoint_updates=2,
        model_updates=2,
        log_updates=1,
        timeout=60,
    )
    directory = tmp_path / "target"
    run = TrainingRun(config, system, directory, components=("representation", "dynamics"))
    assert run.learner.updates == run.learner.samples == run.positions == run.games == 0
    assert not run.replay.games and not run.learner.optimizer.state
    assert all(p.requires_grad for p in run.learner.model.parameters())
    initial = load_checkpoint(directory / "models/step_00000000.pt")
    assert initial["initialization"]["source_sha256"] == fingerprint(source_path)["sha256"]
    assert initial["initialization"]["source_progress"]["updates"] == 2
    assert initial["progress"]["learner_samples"] == 0
    # Interrupt once with accepted work, then resume with the saved credit and RNG.
    original = TrainingRun.update

    def interrupt_after_update(self):
        original(self)
        self.stop_requested = True

    with monkeypatch.context() as patch:
        patch.setattr(TrainingRun, "update", interrupt_after_update)
        partial = run.run()
    checkpoint = directory / "checkpoints" / f"step_{partial['updates']:08d}.pt"
    result = train(config, system, directory, checkpoint)
    assert result["stopping_reason"] == "position_budget"
    assert result["collection_complete"]
    assert result["positions"] >= 60
    assert result["positions"] < 60 + 2 * config.max_moves
    assert result["games"] == result["next_game"]
    assert result["learner_samples"] <= config.position_ratio * result["positions"]
    assert (
        result["learner_samples"] + config.batch_size > config.position_ratio * result["positions"]
    )
    assert result["position_budget_overshoot"] == result["positions"] - 60
    final = directory / "models/final.pt"
    saved = load_checkpoint(final)
    assert saved["progress"]["positions"] == result["positions"]
    assert saved["progress"]["learner_samples"] == result["learner_samples"]
    assert saved["initialization"] == initial["initialization"]
    model, loaded, progress = load_candidate(final, "cpu")
    assert loaded == config and progress["updates"] == result["updates"]
    match = evaluate(final, final, games=4, batch=4, simulations=4)
    assert match["pair_scores"] == [0.5, 0.5]
    monitored = System(
        evaluations=(
            Evaluation("final", opponent=str(final), games=4, simulations=4, interval_updates=1000),
        )
    )
    evaluations = watch_once(directory, monitored, finished=True)
    assert evaluations[0]["candidate"]["sha256"] == fingerprint(final)["sha256"]
    assert not watch_once(directory, monitored, finished=True)
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["reason"] == "position_budget"
    assert list((directory / "history").glob("*/models/final.pt"))


def test_position_budget_validation_and_retry(tmp_path, monkeypatch):
    config = small(
        schedule="async",
        samples_per_position=0.3,
        warmup_positions=4,
        position_lead=14,
        max_positions=50,
        updates=1000,
        replay_games=1,
    )
    for bad in (
        dataclasses.replace(config, max_positions=3),
        dataclasses.replace(config, max_positions=13),
        dataclasses.replace(config, samples_per_position=None),
        dataclasses.replace(config, max_positions=14.5),
    ):
        with pytest.raises(ValueError, match="position"):
            bad.validate()
    original = TrainingRun._dispatch
    killed = []

    def dispatch(self, limit):
        original(self, limit)
        if self.jobs and not killed:
            peer = next(iter(self.jobs.values()))["peer"]
            peer.process.terminate()
            killed.append(peer)

    monkeypatch.setattr(TrainingRun, "_dispatch", dispatch)
    result = train(config, System(actor_devices=("cpu",), actor_batch=2, timeout=60), tmp_path)
    assert result["games"] == result["next_game"] and result["stopping_reason"] == "position_budget"
    assert (
        result["learner_samples"]
        == int(config.position_ratio * result["positions"] // config.batch_size) * config.batch_size
    )


def test_resume_archives_final_evaluation_even_at_same_update(tmp_path):
    from zero_systems.storage import Journal, atomic_json

    journal = Journal(tmp_path)
    journal.write("learn", updates=3, seconds=5)
    final = tmp_path / "models/final.pt"
    final.parent.mkdir()
    final.write_bytes(b"snapshot")
    evaluation = tmp_path / "evaluations/anchor/final.json"
    atomic_json(evaluation, {"candidate": {"path": str(final), "progress": {"updates": 3}}})
    journal.recover({"updates": 3, "seconds": 5})
    journal.close()
    assert not final.exists() and not evaluation.exists()
    assert list((tmp_path / "history").glob("*/evaluations/anchor/final.json"))


@pytest.mark.cuda
@pytest.mark.parametrize("network", ["residual", "transformer"])
def test_spatial_compiled_bf16(network):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from zero_systems.distributed import setup_process
    from zero_systems.env import make_environment
    from zero_systems.play import play_games
    from zero_systems.replay import Replay, make_batch
    from zero_systems.search import Search

    config = small(network=network, prediction_head="spatial")
    system = System(learner_devices=("cuda:0",), compile=True, precision="bfloat16")
    setup_process(system, "cuda:0", 303)
    env = make_environment(config, 0)
    model = build_model(config, env.task).to("cuda:0")
    search = Search(model, config, "cuda:0", "bfloat16", compile=True)
    for count in (4, 3, 1):
        result = search([env.position()] * count, [np.random.default_rng(i) for i in range(count)])
        assert np.isfinite(result.policy).all()
    replay = Replay(8, env.task)
    for game in play_games(Search(model, config, "cuda:0", "bfloat16"), config, [0, 1], 0):
        replay.add(game)
    learner = Learner(model, config, system, "cuda:0")
    for _ in range(2):
        metrics = learner.step(make_batch(replay, config, np.random.default_rng(0)))
        assert np.isfinite(metrics["loss"]) and np.isfinite(metrics["grad_norm"])
    env.close()


@pytest.mark.cuda
@pytest.mark.parametrize("ranks,microbatch", [(1, 7), (4, 2)])
def test_position_budget_counts_global_samples_on_gpus(tmp_path, ranks, microbatch):
    if torch.cuda.device_count() < ranks:
        pytest.skip("requested GPU count unavailable")
    config = small(
        prediction_head="spatial",
        schedule="async",
        updates=1000,
        batch_size=7,
        samples_per_position=0.7,
        warmup_positions=10,
        position_lead=20,
        max_positions=32,
        max_moves=2,
    )
    system = System(
        learner_devices=tuple(f"cuda:{i}" for i in range(ranks)),
        actor_devices=("cpu",),
        actor_batch=2,
        microbatch=microbatch,
        timeout=90,
    )
    result = train(config, system, tmp_path)
    assert result["stopping_reason"] == "position_budget"
    assert result["learner_samples"] == result["updates"] * 7
    assert result["learner_samples"] == int(config.position_ratio * result["positions"] // 7) * 7
    assert result["learner_samples"] == 21
