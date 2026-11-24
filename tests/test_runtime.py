import dataclasses
import json
import socket
import struct
import threading
import time

import numpy as np
import pytest
import torch
from test_core import small

from zero_systems.config import System
from zero_systems.run import AcceptedGames, TrainingRun, train
from zero_systems.storage import decode_game, encode_game, source_identity
from zero_systems.workers import actor_node


def load_run(folder, update):
    return torch.load(folder / "checkpoints" / f"step_{update:08d}.pt", weights_only=False)


def test_checkpoint_survives_package_rename(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    from zero_systems import data
    from zero_systems.evaluate import load_candidate
    from zero_systems.storage import load_checkpoint

    config, system = small(updates=2), System(actor_batch=4)
    train(config, system, tmp_path / "original")
    record = load_run(tmp_path / "original", 2)
    record["kind"] = "v3-full"
    checkpoint = tmp_path / "before-rename.pt"
    with monkeypatch.context() as old_names:
        package = ModuleType("zero_systems_v3")
        package.data = data
        old_names.setitem(sys.modules, "zero_systems_v3", package)
        old_names.setitem(sys.modules, "zero_systems_v3.data", data)
        for kind in (data.Batch, data.Trajectory):
            old_names.setattr(kind, "__module__", "zero_systems_v3.data")
        torch.save(record, checkpoint)
    assert "zero_systems_v3" not in sys.modules
    restored = load_checkpoint(checkpoint)
    assert isinstance(restored["replay"][0], data.Trajectory)
    assert restored["kind"] == "full"
    _, _, progress = load_candidate(checkpoint, "cpu")
    assert progress["updates"] == 2
    result = train(dataclasses.replace(config, updates=4), system, tmp_path / "resumed", checkpoint)
    assert result["updates"] == 4
    assert result["learner_samples"] == 4 * config.batch_size
    uninterrupted = tmp_path / "uninterrupted"
    train(dataclasses.replace(config, updates=4), system, uninterrupted)
    expected, actual = load_run(uninterrupted, 4), load_run(tmp_path / "resumed", 4)
    assert all(torch.equal(expected["model"][k], actual["model"][k]) for k in expected["model"])


def test_trajectory_wire_roundtrip_and_bounds(monkeypatch):
    from zero_systems import storage
    from zero_systems.data import TaskSpec, Trajectory

    game = Trajectory(
        7,
        13,
        np.arange(24, dtype=">f4").reshape(3, 8)[:, ::2],
        np.empty((3, 0), np.float32),
        np.zeros(3, np.int8),
        np.arange(3, dtype=np.int16) % 2,
        np.arange(3, dtype=np.float64)[:, None],
        np.full((3, 2), 0.5, np.float32),
        np.linspace(0, 1, 3, dtype=np.float32),
        21,
        False,
        True,
        bootstrap=0.25,
    )
    task = TaskSpec((4,), 2, 1)
    game.validate(task)
    payload = encode_game(game)
    restored = decode_game(payload)
    restored.validate(task)
    for key, expected in vars(game).items():
        actual = getattr(restored, key)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected, strict=True)
            assert not actual.flags.writeable
        else:
            assert actual == expected
    for invalid in (payload[:7], payload[:-1], payload + b"x", b"BAD!" + payload[4:]):
        with pytest.raises(ValueError):
            decode_game(invalid)
    size = struct.unpack_from("<I", payload, 4)[0]
    for dtype, shape in (("O", [3, 4]), (">f4", [-1, 4]), (">f4", [2**50])):
        header = json.loads(payload[8 : 8 + size])
        header["arrays"][0][1:] = [dtype, shape]
        encoded = json.dumps(header).encode()
        invalid = struct.pack("<4sI", b"ZSG4", len(encoded)) + encoded + payload[8 + size :]
        with pytest.raises(ValueError):
            decode_game(invalid)
    with pytest.raises(ValueError, match="numeric"):
        encode_game(dataclasses.replace(game, observations=game.observations.astype(object)))
    monkeypatch.setattr(storage, "MAX_PAYLOAD", len(payload) - 1)
    with pytest.raises(ValueError, match="size"):
        decode_game(payload)
    with pytest.raises(ValueError, match="large"):
        encode_game(game)


def test_resume_preserves_mid_round_state(tmp_path):
    cfg = small(updates=3, resign_after=0, resign_threshold=2.0, no_resign_fraction=1.0)
    system = System(actor_batch=4, microbatch=2, model_updates=2, log_updates=1)
    complete, resumed = tmp_path / "complete", tmp_path / "resumed"
    train(cfg, system, complete)
    train(dataclasses.replace(cfg, updates=1), system, resumed)
    checkpoint = resumed / "checkpoints" / "step_00000001.pt"
    old_record = load_run(resumed, 1)
    assert old_record["pending_batch"] is not None
    old_record["experiment"].pop("value_bottleneck_slope")
    old_record["experiment"].pop("value_bottleneck_bias")
    torch.save(old_record, checkpoint)
    train(cfg, system, resumed, checkpoint)
    a, b = load_run(complete, 3), load_run(resumed, 3)
    assert a["sampling_rng"] == b["sampling_rng"]
    for key, value in vars(a["pending_batch"]).items():
        torch.testing.assert_close(value, getattr(b["pending_batch"], key), rtol=0, atol=0)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    for key in (
        "games",
        "positions",
        "learner_samples",
        "phase",
        "remaining_updates",
        "resigned_games",
        "resignation_tests",
        "false_resignations",
    ):
        assert a["progress"][key] == b["progress"][key]
    assert a["progress"]["resignation_tests"] == a["progress"]["games"]
    assert a["progress"]["false_resignations"] > 0
    with pytest.raises(ValueError, match="directory already"):
        train(cfg, system, complete)
    train(cfg, system, resumed, checkpoint)
    assert list((resumed / "history").glob("*/models/step_00000003.pt"))
    rollback = load_run(resumed, 3)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], rollback["model"][key], rtol=0, atol=0)
    trajectory = a["replay"][0]
    decoded = decode_game(encode_game(trajectory))
    for key, value in vars(trajectory).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(decoded, key))
        else:
            assert value == getattr(decoded, key)
    accepted = AcceptedGames()
    assert accepted.accept(1) and accepted.accept(0)
    assert accepted.prefix == 2 and not accepted.accept(1)


@pytest.mark.parametrize("schedule", ["phased", "async"])
def test_local_worker_complete_training(tmp_path, schedule):
    cfg = small(updates=6 if schedule == "async" else 2, schedule=schedule, publish_updates=2)
    system = System(
        actor_devices=("cpu",), actors_per_device=2, actor_batch=2, microbatch=2, timeout=60
    )
    result = train(cfg, system, tmp_path)
    assert result["games"] == 2 * cfg.updates
    assert result["learner_samples"] == 4 * cfg.updates
    if schedule == "async":
        games = load_run(tmp_path, cfg.updates)["replay"]
        assert max(game.generating_update for game in games) >= 2


def test_worker_death_retries_only_missing_games(tmp_path, monkeypatch):
    original = TrainingRun._dispatch
    killed = []

    def dispatch(run, limit):
        original(run, limit)
        if not killed and run.jobs:
            peer = next(iter(run.jobs.values()))["peer"]
            peer.process.terminate()
            killed.append(peer)

    monkeypatch.setattr(TrainingRun, "_dispatch", dispatch)
    result = train(
        small(updates=2), System(actor_devices=("cpu",), actor_batch=4, timeout=60), tmp_path
    )
    record = load_run(tmp_path, 2)
    assert result["games"] == result["next_game"] == 4
    assert len({game.game_id for game in record["replay"]}) == 4


def test_board_transfer_and_new_context_are_explicit():
    from zero_systems.env import task_spec
    from zero_systems.model import build_model, initialize

    config = small()
    source_task = task_spec(config)
    source = build_model(config, source_task)
    larger = dataclasses.replace(config, board_size=5, condition_komi=True, komi=(1.0, 2.0))
    target = build_model(larger, task_spec(larger))
    report = initialize(target, source.state_dict(), source_task=source_task)
    assert report["expanded"] == ["representation.0.0.weight"]
    assert "prediction.policy.2.weight" in report["reset"]
    assert target.representation[0][0].weight[:, -1].count_nonzero() == 0
    torch.testing.assert_close(target.dynamics[0][0].weight, source.dynamics[0][0].weight)


def test_paired_arena_swaps_colors(tmp_path):
    from zero_systems.evaluate import evaluate

    train(small(updates=2), System(actor_batch=4), tmp_path)
    model = tmp_path / "models" / "step_00000002.pt"
    result = evaluate(model, model, games=4, batch=4, simulations=4)
    assert result["score"] == 0.5
    assert result["pair_scores"] == [0.5, 0.5]


def test_remote_wire_roundtrip(tmp_path, monkeypatch):
    from multiprocessing import AuthenticationError
    from multiprocessing.connection import Client

    monkeypatch.setenv("ZERO_SYSTEMS_AUTHKEY", "local-integration-test")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    address = f"127.0.0.1:{port}"
    errors = []

    def connect():
        rejected = False
        for _ in range(100):
            try:
                if not rejected:
                    try:
                        Client(("127.0.0.1", port), authkey=b"wrong-secret")
                    except AuthenticationError:
                        rejected = True
                actor_node(
                    address, System(actor_batch=2, actor_devices=("cpu",), actors_per_device=2)
                )
                return
            except ConnectionRefusedError:
                time.sleep(0.05)
            except BaseException as error:
                errors.append(error)
                return
        errors.append(TimeoutError("remote server did not start"))

    thread = threading.Thread(target=connect, daemon=True)
    thread.start()
    config = small(updates=2, resign_after=0, resign_threshold=2.0, no_resign_fraction=1.0)
    result = train(config, System(listen=address, timeout=60), tmp_path)
    thread.join(timeout=5)
    assert not thread.is_alive() and not errors
    assert result["games"] == 4
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    (remote,) = [r for r in records if r["event"] == "remote_actor_connected"]
    assert remote["actor_source_sha256"] == source_identity()
    assert remote["hostname"] == socket.gethostname()
    assert result["remote_games"] == 4
    assert result["remote_positions"] == result["positions"]
    assert result["resignation_tests"] == 4 and result["resigned_games"] == 0


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs unavailable")
def test_ddp_uneven_batch(tmp_path):
    cfg = small(updates=2, batch_size=7)
    system = System(learner_devices=("cuda:0", "cuda:1"), microbatch=2, actor_batch=4, timeout=90)
    result = train(cfg, system, tmp_path)
    assert result["learner_samples"] == 14


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs unavailable")
def test_ddp_matches_global_mean_without_batchnorm():
    import copy

    from zero_systems.data import Batch, TaskSpec
    from zero_systems.distributed import LearnerGroup
    from zero_systems.learn import Learner
    from zero_systems.model import build_model
    from zero_systems.storage import weight_bytes

    torch.set_num_threads(1)
    torch.manual_seed(123)
    config = small(
        environment="CartPole-v1",
        network="mlp",
        augmentation=False,
        reward_weight=1,
        batch_size=7,
        grad_clip=1e6,
        momentum=0,
        weight_decay=0,
        learning_rate=0.01,
    )
    task = TaskSpec((4,), 2, 1)
    model = build_model(config, task)
    batch = Batch(
        torch.randn(7, 4),
        torch.empty(7, 0),
        torch.randint(2, (7, 2)),
        torch.randn(7, 3, 2).softmax(-1),
        torch.randn(7, 3),
        torch.randn(7, 3),
        torch.ones(7, 3),
        torch.ones(7, 3),
        torch.arange(7),
        torch.zeros(7, dtype=torch.int64),
    )
    single = Learner(copy.deepcopy(model), config, System(microbatch=7), "cuda:0")
    group = LearnerGroup(
        model,
        config,
        System(learner_devices=("cuda:0", "cuda:1"), microbatch=2, timeout=90),
        task,
        weight_bytes(model),
    )
    try:
        for iteration in range(3):
            batch.observation = torch.randn(7, 4) + iteration
            batch.value = torch.randn(7, 3) + iteration
            expected = single.step(batch)
            actual = group.step(batch)
            assert abs(actual["loss"] - expected["loss"]) < 1e-5
            for key, value in single.model.state_dict().items():
                torch.testing.assert_close(
                    group.learner.model.state_dict()[key], value, atol=2e-6, rtol=2e-5
                )
    finally:
        group.close()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("slope,bias", [(0.0, True), (0.1, True), (0.0, False)])
def test_bf16_compiled_search_and_update(slope, bias):
    from zero_systems.distributed import setup_process
    from zero_systems.env import make_environment
    from zero_systems.learn import Learner
    from zero_systems.model import build_model
    from zero_systems.play import play_games
    from zero_systems.replay import Replay, make_batch
    from zero_systems.search import Search

    config = small(value_bottleneck_slope=slope, value_bottleneck_bias=bias)
    system = System(
        learner_devices=("cuda:0",), precision="bfloat16", actor_precision="bfloat16", compile=True
    )
    setup_process(system, "cuda:0", 303)
    env = make_environment(config, 0)
    model = build_model(config, env.task).to("cuda:0")
    search = Search(model, config, "cuda:0", "bfloat16", compile=True)
    for count in (4, 3, 2, 1, 3):
        result = search([env.position()] * count, [np.random.default_rng(i) for i in range(count)])
        assert np.isfinite(result.policy).all()
    replay = Replay(8, env.task)
    # Full episodes use the already validated eager actor to isolate compiled gradients.
    for game in play_games(Search(model, config, "cuda:0", "bfloat16"), config, [0, 1], 0):
        replay.add(game)
    learner = Learner(model, config, system, "cuda:0")
    for _ in range(2):
        result = learner.step(make_batch(replay, config, np.random.default_rng(0)))
        assert np.isfinite(result["loss"])
