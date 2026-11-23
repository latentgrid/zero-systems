import dataclasses
import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from test_core import small

from zero_systems.config import Evaluation, System, from_dict, learning_settings
from zero_systems.env import make_environment
from zero_systems.evaluate import evaluate, evaluate_episodes, load_candidate
from zero_systems.live import watch_once
from zero_systems.run import train


@pytest.mark.parametrize("search", ["puct", "gumbel"])
@pytest.mark.parametrize(
    "environment,kwargs",
    [
        ("CartPole-v1", {}),
        ("FrozenLake-v1", {"is_slippery": False}),
        ("Taxi-v4", {}),
        ("spiel:tic_tac_toe", {}),
        ("spiel:connect_four", {"rows": 4, "columns": 4}),
        ("spiel:cliff_walking", {"height": 3, "width": 4, "horizon": 6}),
    ],
)
def test_general_environment_workflow(tmp_path, environment, kwargs, search):
    config = small(
        environment=environment,
        environment_kwargs=kwargs,
        network="mlp",
        augmentation=False,
        reward_weight=1,
        value_perspective="to_play",
        search=search,
        updates=2,
        max_moves=12,
    )
    result = train(config, System(actor_batch=4), tmp_path)
    checkpoint = tmp_path / "models/step_00000002.pt"
    model, _, _ = load_candidate(checkpoint, "cpu")
    if model.task.players == 1:
        evaluation = evaluate_episodes(checkpoint, episodes=4, simulations=4)
    else:
        evaluation = evaluate(checkpoint, checkpoint, games=4, simulations=4, opening_moves=0)
        assert evaluation["score"] == 0.5
    assert np.isfinite(evaluation["mean_return"])
    assert result["learner_samples"] == 8
    if environment == "spiel:cliff_walking":
        saved = torch.load(tmp_path / "checkpoints/step_00000002.pt", weights_only=False)
        assert any(np.any(game.rewards < 0) for game in saved["replay"])


class Masked(gym.Env):
    observation_space = gym.spaces.Dict(
        {
            "features": gym.spaces.Tuple(
                (gym.spaces.Box(-1, 1, (2,), dtype=np.float32), gym.spaces.Discrete(3))
            ),
            "mask": gym.spaces.MultiBinary(2),
        }
    )
    action_space = gym.spaces.Discrete(2, start=7)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {"features": (np.array([0.5, -0.5], np.float32), 2), "mask": np.array([0, 1])}, {}

    def step(self, action):
        assert action == 8
        observation, info = self.reset()
        return observation, 7.5, np.bool_(False), np.bool_(True), info


def test_structured_observations_masks_and_action_translation(monkeypatch):
    def make(name, **kwargs):
        assert kwargs == {"example": 3}
        return Masked()

    monkeypatch.setattr(gym, "make", make)
    config = small(
        environment="custom",
        environment_kwargs={"example": 3},
        network="mlp",
        augmentation=False,
        action_mask_source="observation",
        action_mask_key="mask",
    )
    env = make_environment(config, 3)
    np.testing.assert_array_equal(env.position().observation, [0.5, -0.5, 0, 0, 1])
    np.testing.assert_array_equal(env.position().legal, [False, True])
    assert env.step(1)[0] == 7.5 and env.truncated and not env.terminated
    assert type(env.terminated) is bool and type(env.truncated) is bool
    env.observation["mask"] = [0, 0]
    with pytest.raises(ValueError, match="mask"):
        env.position()
    del env.observation["mask"]
    with pytest.raises(ValueError, match="missing"):
        env.position()
    env.close()


@pytest.mark.parametrize("game", ["kuhn_poker", "pig", "matrix_pd"])
def test_unsupported_spiel_contract(game):
    with pytest.raises(ValueError, match="OpenSpiel adapter"):
        make_environment(small(environment="spiel:" + game), 0)


def test_position_credit_and_legacy_ratio():
    config = small(
        schedule="async",
        samples_per_position=0.3,
        batch_size=4,
        warmup_positions=3,
        position_lead=14,
    )
    config.validate()
    assert not config.can_update(9, 13, 0, 0)
    assert config.can_update(9, 14, 0, 0)
    assert not config.can_update(9, 26, 4, 1)
    assert config.can_update(9, 27, 4, 1)
    for invalid in (
        dataclasses.replace(config, games_per_update=2),
        dataclasses.replace(config, position_lead=13),
    ):
        with pytest.raises(ValueError):
            invalid.validate()
    old = small(schedule="async")
    new = dataclasses.replace(old, games_per_update=2)
    assert learning_settings(old) == learning_settings(new)
    assert from_dict(
        dataclasses.asdict(System(evaluations=(Evaluation("test"),))), System
    ).evaluations == (Evaluation("test"),)


def test_position_scheduler_resume_and_replay_eviction(tmp_path):
    config = small(
        schedule="async",
        samples_per_position=0.3,
        warmup_positions=4,
        position_lead=14,
        replay_games=1,
        updates=2,
    )
    system = System(actor_devices=("cpu",), actor_batch=2, timeout=60, log_updates=1)
    first = train(config, system, tmp_path)
    checkpoint = tmp_path / "checkpoints/step_00000002.pt"
    second = train(dataclasses.replace(config, updates=4), system, tmp_path, checkpoint)
    assert second["learner_samples"] == 16 and second["positions"] >= first["positions"]
    assert second["games"] == second["next_game"]
    assert second["learner_samples"] <= float(config.position_ratio) * second["positions"]
    record = torch.load(tmp_path / "checkpoints/step_00000004.pt", weights_only=False)
    assert len(record["replay"]) == 1
    assert record["progress"]["position_lead_overshoot"] == second["position_lead_overshoot"]


def test_tensorboard_rollback_matches_jsonl(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    config = small(updates=2)
    system = System(actor_batch=4, log_updates=1, model_updates=1, tensorboard=True)
    train(config, system, tmp_path)
    train(
        dataclasses.replace(config, updates=4),
        system,
        tmp_path,
        tmp_path / "checkpoints/step_00000002.pt",
    )
    train(
        dataclasses.replace(config, updates=3),
        system,
        tmp_path,
        tmp_path / "checkpoints/step_00000002.pt",
    )
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert max(row.get("updates", 0) for row in rows) == 3
    assert list((tmp_path / "history").glob("*/models/step_00000004.pt"))
    events = EventAccumulator(str(tmp_path / "tensorboard/training")).Reload()
    values = events.Scalars("progress/learner_samples")
    assert max(value.step for value in values) == 3 and values[-1].value == 12
    saved = [r for r in rows if r["event"] == "model_available"]
    assert saved and all(r["model_save_seconds"] >= 0 for r in saved)
    assert all(r["monotonic_time"] > 0 for r in saved)
    assert events.Scalars("timing/model_save_seconds")
    assert events.Scalars("learning/target_policy_entropy")


def test_live_final_lag_dedup_and_protocol_change(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    train(small(updates=2), System(actor_batch=4), tmp_path)
    system = System(
        tensorboard=True,
        evaluations=(
            Evaluation("training", games=4, simulations=4, interval_updates=10),
            Evaluation("lag", mode="self_lag", lag_updates=1, games=4, simulations=4),
            Evaluation("missing", mode="self_lag", lag_updates=10, games=4, simulations=4),
        ),
    )
    results = watch_once(tmp_path, system, finished=True)
    assert [r["status"] for r in results] == ["complete", "complete", "skipped"]
    assert results[1]["opponent"]["progress"]["updates"] == 0
    events = EventAccumulator(str(tmp_path / "tensorboard/evaluations/training")).Reload()
    assert events.Scalars("evaluation/training/score")[0].step == 2
    assert not (tmp_path / "tensorboard/training").exists()
    assert not [r for r in watch_once(tmp_path, system, finished=True) if r["status"] == "complete"]
    changed = dataclasses.replace(
        system, evaluations=(dataclasses.replace(system.evaluations[0], seed=2),)
    )
    assert watch_once(tmp_path, changed, finished=True)[0]["status"] == "complete"


def test_live_failure_does_not_stop_other_jobs(tmp_path, monkeypatch):
    from zero_systems import live

    train(small(updates=2), System(actor_batch=4), tmp_path)
    original = live.evaluate

    def evaluate_or_fail(*args, **kwargs):
        if kwargs["seed"] == 1:
            raise RuntimeError("injected evaluation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(live, "evaluate", evaluate_or_fail)
    system = System(
        evaluations=(
            Evaluation("bad", seed=1, games=4, simulations=4),
            Evaluation("absent", opponent=str(tmp_path / "missing.pt"), games=4, simulations=4),
            Evaluation("good", games=4, simulations=4),
        )
    )
    results = watch_once(tmp_path, system, finished=True)
    assert [r["status"] for r in results] == ["failed", "failed", "complete"]


def test_automatic_live_process_evaluates_final(tmp_path):
    system = System(
        actor_batch=4,
        evaluations=(Evaluation("random", games=4, simulations=4, interval_updates=100),),
    )
    result = train(small(updates=2), system, tmp_path)
    records = [json.loads(p.read_text()) for p in (tmp_path / "evaluations/random").glob("*.json")]
    assert len(records) == 1 and records[0]["status"] == "complete"
    assert records[0]["candidate"]["progress"]["learner_samples"] == result["learner_samples"]
    assert result["evaluations"]["random"]["status"] == "complete"
    assert result["samples_per_position"] == result["learner_samples"] / result["positions"]


def test_evaluator_process_death_preserves_training(tmp_path, monkeypatch):
    import multiprocessing as mp
    import os

    from zero_systems import live

    def start_dying_evaluator(*_):
        context = mp.get_context("spawn")
        process = context.Process(target=os._exit, args=(77,))
        process.start()
        return process, context.Event()

    monkeypatch.setattr(live, "start_watcher", start_dying_evaluator)
    result = train(small(updates=2), System(evaluations=(Evaluation("random"),)), tmp_path)
    assert result["updates"] == 2
    assert result["evaluations"]["watcher"]["status"] == "failed"
    assert "77" in result["evaluations"]["watcher"]["error"]
    assert (tmp_path / "models/step_00000002.pt").exists()


def test_evaluation_rejects_rule_changes_and_mutable_snapshots(tmp_path, monkeypatch):
    from zero_systems import evaluate as arena

    train(small(updates=2), System(actor_batch=4), tmp_path)
    candidate = tmp_path / "models/step_00000002.pt"
    record = torch.load(candidate, weights_only=False)
    record["experiment"]["komi"] = (2.5,)
    other = tmp_path / "different-rules.pt"
    torch.save(record, other)
    with pytest.raises(ValueError, match="rules differ"):
        evaluate(candidate, other, games=4, simulations=4)
    original = arena._episodes

    def mutate_after_play(*args, **kwargs):
        result = original(*args, **kwargs)
        changed = torch.load(candidate, weights_only=False)
        changed["progress"]["seconds"] += 1
        torch.save(changed, candidate)
        return result

    monkeypatch.setattr(arena, "_episodes", mutate_after_play)
    with pytest.raises(RuntimeError, match="changed during"):
        evaluate(candidate, "random", games=4, simulations=4)


def test_slow_live_evaluation_finds_new_final_checkpoint(tmp_path, monkeypatch):
    from zero_systems import live

    train(small(updates=2), System(actor_batch=4), tmp_path)
    original = live.evaluate

    def publish_during_evaluation(*args, **kwargs):
        result = original(*args, **kwargs)
        record = torch.load(tmp_path / "models/step_00000002.pt", weights_only=False)
        record["progress"]["updates"] = 4
        torch.save(record, tmp_path / "models/step_00000004.pt")
        return result

    monkeypatch.setattr(live, "evaluate", publish_during_evaluation)
    system = System(evaluations=(Evaluation("random", interval_updates=2, games=4, simulations=4),))
    first = watch_once(tmp_path, system)
    assert first[0]["checkpoint_age_updates"] == 2
    monkeypatch.setattr(live, "evaluate", original)
    final = watch_once(tmp_path, system, finished=True)
    assert final[0]["candidate"]["progress"]["updates"] == 4


def test_checkpoint_rejects_changed_gym_encoding(tmp_path):
    config = small(
        environment="FrozenLake-v1",
        environment_kwargs={"is_slippery": False},
        network="mlp",
        augmentation=False,
        reward_weight=1,
        updates=2,
    )
    train(config, System(actor_batch=4), tmp_path)
    model = tmp_path / "models/step_00000002.pt"
    record = torch.load(model, weights_only=False)
    record["experiment"]["environment_kwargs"]["is_slippery"] = True
    torch.save(record, model)
    with pytest.raises(ValueError, match="encoding"):
        load_candidate(model, "cpu")
