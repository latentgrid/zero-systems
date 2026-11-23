import dataclasses

import numpy as np
import pytest
import torch

from zero_systems.config import Experiment, System
from zero_systems.data import TaskSpec, Trajectory, game_seed
from zero_systems.env import make_environment, task_spec
from zero_systems.learn import Learner, scaled_gradient, shard_bounds
from zero_systems.model import build_model
from zero_systems.play import play_games
from zero_systems.replay import Replay, make_batch, value_targets
from zero_systems.search import Search, SearchResult, halving_visits


def small(**kwargs):
    values = dict(
        board_size=3,
        history=2,
        channels=8,
        blocks=1,
        value_hidden=8,
        max_moves=19,
        simulations=4,
        considered_actions=4,
        batch_size=4,
        unroll=2,
        games_per_round=4,
        updates_per_round=2,
        warmup_games=4,
        replay_games=8,
        game_lead=4,
        updates=4,
        resign_after=1000,
    )
    values.update(kwargs)
    return Experiment(**values)


def test_configuration_rejects_deadlocks():
    with pytest.raises(ValueError, match="warmup"):
        small(schedule="async", game_lead=2).validate()
    with pytest.raises(ValueError, match="conditioning"):
        small(komi=(1.0, 2.0)).validate()
    with pytest.raises(ValueError, match="warmup"):
        small(warmup_games=6).validate()
    with pytest.raises(ValueError, match="finite"):
        System(timeout=float("nan")).validate()
    with pytest.raises(ValueError, match="bottleneck"):
        small(value_bottleneck_slope=1.1).validate()
    with pytest.raises(ValueError, match="bottleneck"):
        small(network="mlp", value_bottleneck_slope=0.1).validate()
    with pytest.raises(ValueError, match="bottleneck"):
        small(network="mlp", value_bottleneck_bias=False).validate()
    assert game_seed(3, 42) == game_seed(3, 42)
    assert game_seed(3, 42) != game_seed(3, 42, 1)


def test_go_history_is_relative_to_current_player():
    env = make_environment(small(), 0)
    assert env.task.encoding == "go-relative-history-v1"
    env.step(0)
    white = env.position().observation
    assert white[0].sum() == 0 and white[1, 0, 0] == 1
    assert white[-1].sum() == 9 and white[-2].sum() == 0
    env.step(1)
    black = env.position().observation
    assert black[0, 0, 0] == 1 and black[1, 0, 1] == 1
    assert black[2, 0, 0] == 1 and black[3].sum() == 0
    assert black[-2].sum() == 9 and black[-1].sum() == 0


def test_go_superko_mask_preserves_move_limit_draws():
    history = [7, 1, 3, 0, 2, 8, 4, 1, 9, 5, 2]
    for limit, komi, moves in ((19, 7.0, history), (12, 7.0, history), (2, 0.0, [0])):
        env = make_environment(small(max_moves=limit, komi=(komi,)), 0)
        for action in moves:
            env.step(action)
        mask = env.position().legal
        # Independent exact oracle, including the ambiguous final-move boundary.
        for action in env.state.legal_actions():
            child = env.state.clone()
            child.apply_action(action)
            expected = action == 9 or env._board(child).tobytes() not in env.seen
            assert mask[action] == expected
        assert 8 in env.state.legal_actions()
        if limit == 2:
            assert mask[8]
            env.step(8)
            np.testing.assert_array_equal(env.outcome, [0, 0])
        else:
            assert not mask[8]
        env.close()


def test_gradient_boundary():
    x = torch.tensor(2.0, requires_grad=True)
    prediction = x * x
    outgoing = scaled_gradient(x, 0.5)
    (prediction + 6 * outgoing).backward()
    assert x.grad == 7


@pytest.mark.parametrize("slope,bias", [(0.0, True), (0.1, True), (0.0, False)])
def test_value_bottleneck_retains_negative_feature_gradients(slope, bias):
    torch.manual_seed(31)
    config = small(value_bottleneck_slope=slope, value_bottleneck_bias=bias)
    model = build_model(config, task_spec(config))
    head = model.prediction["value"]
    norm = head[0][1]
    with torch.no_grad():
        norm.weight.fill_(0.02)
        if norm.bias is not None:
            norm.bias.fill_(-0.8)
    # A learned negative offset deactivates every feature in this batch.
    value = head(torch.rand(32, config.channels, 3, 3))
    (value - 1).square().mean().backward()
    assert bool(norm.weight.grad.abs().sum() > 0) == (slope > 0 or not bias)
    if norm.bias is not None:
        assert bool(norm.bias.grad.abs().sum() > 0) == (slope > 0)


def test_fixed_value_offset_preserves_initial_predictions():
    original_config = small()
    fixed_config = small(value_bottleneck_bias=False)
    torch.manual_seed(31)
    original = build_model(original_config, task_spec(original_config)).eval()
    torch.manual_seed(31)
    fixed = build_model(fixed_config, task_spec(fixed_config)).eval()
    for key, value in fixed.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
    observation = torch.rand(4, *task_spec(fixed_config).observation_shape)
    context = torch.empty(4, 0)
    with torch.no_grad():
        expected = original.initial_inference(observation, context)
        actual = fixed.initial_inference(observation, context)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_replay_symmetry_keeps_actions_and_policy_aligned():
    config = small(unroll=2)
    task = task_spec(config)
    # Plane k marks the action to be taken at unroll k. Its transformed location
    # must agree with both the target policy and the dynamics input action.
    observation = np.zeros((3, *task.observation_shape), np.uint8)
    observation[0, 0, 0, 1] = 1
    observation[0, 1, 1, 2] = 1
    policies = np.eye(10, dtype=np.float32)[[1, 5, 9]]
    game = Trajectory(
        0,
        0,
        observation,
        np.empty((3, 0), np.float32),
        np.array([0, 1, 0]),
        np.array([1, 5, 9]),
        np.zeros((3, 2), np.float32),
        policies,
        np.zeros(3, np.float32),
        0,
        True,
        False,
        outcome=np.array([1.0, -1.0]),
    )
    replay = Replay(8, task)
    replay.add(game)
    seen = set()
    for seed in range(64):
        batch = make_batch(replay, config, np.random.default_rng(seed), [(game, 0)])
        markers = batch.observation[0, :2].flatten(1).argmax(1)
        torch.testing.assert_close(batch.actions[0], markers)
        torch.testing.assert_close(batch.policy[0, :2].argmax(1), markers)
        assert batch.policy[0, 2, 9] == 1
        seen.add(tuple(markers.tolist()))
    assert len(seen) == 8

    replay = Replay(2, task)
    assert replay.add(game) and not replay.add(game)
    replay.add(dataclasses.replace(game, game_id=1))
    fields = ("observations", "contexts", "players", "actions", "rewards", "policies", "values")
    replay.add(dataclasses.replace(game, game_id=2, **{k: getattr(game, k)[:1] for k in fields}))
    expected = np.random.default_rng(9).integers(4, size=32)
    sampled = replay.sample(32, np.random.default_rng(9))
    assert [(g.game_id, t) for g, t in sampled] == [
        (1, int(t)) if t < 3 else (2, 0) for t in expected
    ]


@pytest.mark.parametrize("perspective", ["player0", "to_play"])
@pytest.mark.parametrize("ending", ["outcome", "reward", "truncated"])
def test_unroll_ending_keeps_reward_and_value_perspectives(perspective, ending):
    config = small(unroll=3, augmentation=False, value_perspective=perspective, discount=0.5)
    task = TaskSpec((1,), 3, 2)
    rewards = np.array([[1, -1], [2, -2]], np.float32)
    game = Trajectory(
        0,
        0,
        np.zeros((2, 1), np.float32),
        np.empty((2, 0), np.float32),
        np.array([0, 1]),
        np.array([0, 1]),
        rewards * int(ending != "outcome"),
        np.eye(3, dtype=np.float32)[:2],
        np.zeros(2, np.float32),
        0,
        ending != "truncated",
        ending == "truncated",
        bootstrap=0.25,
        bootstrap_player=1,
        outcome=np.array([1, -1], np.float32) if ending == "outcome" else None,
    )
    replay = Replay(2, task)
    replay.add(game)
    batch = make_batch(replay, config, np.random.default_rng(0), [(game, 1)])
    sign = 1 if perspective == "player0" else -1
    if ending == "outcome":
        expected = [0.5, 1, 1, 1] if sign == 1 else [-0.5, 1, -1, 1]
        torch.testing.assert_close(batch.value[0], torch.tensor(expected, dtype=torch.float32))
        assert batch.valid.all() and not batch.reward.any()
        torch.testing.assert_close(batch.policy[0, 1:], torch.full((3, 3), 1 / 3))
    else:
        value = 1.875 if ending == "truncated" else 2.0
        torch.testing.assert_close(batch.value[0], torch.tensor([sign * value, 0, 0, 0]))
        torch.testing.assert_close(batch.reward[0], torch.tensor([0.0, sign * 2.0, 0, 0]))
        if ending == "reward":
            assert batch.valid.all()
            torch.testing.assert_close(batch.policy[0, 1:], torch.full((3, 3), 1 / 3))
        else:
            torch.testing.assert_close(batch.valid[0], torch.tensor([1.0, 0, 0, 0]))
            assert not batch.policy[0, 1:].any()
    assert batch.actions[0, 0] == 1
    torch.testing.assert_close(batch.policy[0, 0], torch.tensor([0.0, 1, 0]))
    expected_mask = [0.0, 1, 1, 1] if ending == "reward" else [0.0, 1, 0, 0]
    torch.testing.assert_close(batch.reward_valid[0], torch.tensor(expected_mask))


@pytest.mark.parametrize("algorithm", ["puct", "gumbel"])
def test_search_backup_changes_perspective_once(algorithm):
    from zero_systems.data import Position

    class Constant(torch.nn.Module):
        task = TaskSpec((1,), 3, 2)

        def initial_inference(self, observation, context):
            return (
                observation.float(),
                torch.zeros(len(observation), 3),
                torch.full((len(observation),), 0.2),
            )

        def recurrent_inference(self, latent, action):
            return (
                latent,
                torch.zeros(len(latent), 3),
                torch.full((len(latent),), 0.8),
                torch.zeros(len(latent)),
            )

    positions = [
        Position(np.zeros(1), np.array([False, True, False]), player, np.empty(0))
        for player in (0, 1)
    ]
    config = small(search=algorithm, simulations=1)
    result = Search(Constant(), config)(positions, [np.random.default_rng(0)] * 2, explore=False)
    np.testing.assert_allclose(result.value, [0.8, -0.8])
    np.testing.assert_allclose(result.action_value, [0.8, -0.8])
    np.testing.assert_array_equal(result.actions, [1, 1])
    np.testing.assert_array_equal(result.policy, [[0, 1, 0]] * 2)


def test_gumbel_greedy_action_uses_search_means():
    from zero_systems.data import Position

    class Branches(torch.nn.Module):
        task = TaskSpec((1,), 2, 1)

        def initial_inference(self, observation, context):
            return (
                observation.float(),
                torch.zeros((len(observation), 2)),
                torch.zeros(len(observation)),
            )

        def recurrent_inference(self, latent, action):
            first = latent[:, 0] == 0
            state = torch.where(first, action.float() + 1, latent[:, 0])[:, None]
            value = torch.where(
                first,
                torch.where(action == 0, 1.0, 0.1),
                torch.where(latent[:, 0] == 1, -1.0, 0.8),
            )
            return state, torch.zeros((len(latent), 2)), value, torch.zeros(len(latent))

    config = small(simulations=16, considered_actions=2)
    position = Position(np.zeros(1), np.ones(2, bool), 0, np.empty(0))
    result = Search(Branches(), config)([position], [np.random.default_rng(0)], explore=False)
    np.testing.assert_array_equal(result.visits, [[8, 8]])
    # The first branch starts optimistically but its later returns are poor.
    assert result.actions[0] == 1
    np.testing.assert_allclose(result.action_value, [0.7125])


def test_resignation_requires_a_losing_selected_action():
    class PassingSearch:
        def __init__(self, action_value):
            self.action_value = action_value

        def __call__(self, positions, rngs, **kwargs):
            count = len(positions)
            policy = np.zeros((count, 10), np.float32)
            policy[:, 9] = 1
            return SearchResult(
                np.full(count, 9),
                policy,
                np.full(count, -0.95),
                policy.astype(np.int32),
                np.full(count, self.action_value),
            )

    config = small(no_resign_fraction=0.0, resign_after=0)
    played = play_games(PassingSearch(0.1), config, [0], 0)[0]
    resigned = play_games(PassingSearch(-0.95), config, [0], 0)[0]
    assert len(played) == 2  # Both players pass; komi gives white the win.
    np.testing.assert_array_equal(played.outcome, [-1, 1])
    assert len(resigned) == 1  # White resigns before the second pass.
    np.testing.assert_array_equal(resigned.outcome, [1, -1])
    assert resigned.resigned_player == 1 and resigned.resignation_test_player == -1
    tested = play_games(
        PassingSearch(-0.95), dataclasses.replace(config, no_resign_fraction=1.0), [0], 0
    )[0]
    assert tested.resignation_test_player == 1 and tested.resigned_player == -1
    assert tested.outcome[tested.resignation_test_player] == 1
    from zero_systems.storage import decode_game, encode_game

    assert decode_game(encode_game(tested)).resignation_test_player == 1


def test_dense_reward_and_truncation_targets():
    cfg = small(
        environment="CartPole-v1",
        augmentation=False,
        network="mlp",
        td_steps=5,
        discount=0.5,
        value_perspective="to_play",
    )
    game = Trajectory(
        0,
        1,
        np.zeros((2, 4), np.float32),
        np.zeros((2, 0), np.float32),
        np.zeros(2, np.int32),
        np.zeros(2, np.int32),
        np.array([[2.0], [4.0]], np.float32),
        np.full((2, 2), 0.5, np.float32),
        np.zeros(2, np.float32),
        0,
        True,
        False,
    )
    np.testing.assert_allclose(value_targets(game, cfg), [4, 4])
    replay = Replay(2, TaskSpec((4,), 2, 1))
    replay.add(game)
    batch = make_batch(replay, cfg, np.random.default_rng(0), [(game, 1)])
    np.testing.assert_array_equal(batch.valid, [[1, 1, 1]])
    np.testing.assert_array_equal(batch.value, [[4, 0, 0]])
    np.testing.assert_array_equal(batch.reward, [[0, 4, 0]])
    np.testing.assert_array_equal(batch.reward_valid, [[0, 1, 1]])
    game = dataclasses.replace(game, terminated=False, truncated=True, bootstrap=8)
    np.testing.assert_allclose(value_targets(game, cfg), [6, 8])


def test_position_sampling_not_game_sampling():
    cfg = small()
    env = make_environment(cfg, 0)
    task = env.task

    def game(identity, length):
        return Trajectory(
            identity,
            0,
            np.zeros((length, *task.observation_shape), np.uint8),
            np.zeros((length, 0), np.float32),
            np.zeros(length, np.int32),
            np.zeros(length, np.int32),
            np.zeros((length, 2), np.float32),
            np.full((length, task.actions), 1 / task.actions, np.float32),
            np.zeros(length, np.float32),
            0,
            True,
            False,
            outcome=np.array([1, -1], np.float32),
        )

    replay = Replay(4, task)
    replay.add(game(0, 1))
    replay.add(game(1, 3))
    samples = replay.sample(12000, np.random.default_rng(19))
    short_fraction = sum(g.game_id == 0 for g, _ in samples) / len(samples)
    assert 0.23 < short_fraction < 0.27


@pytest.mark.parametrize("algorithm", ["gumbel", "puct"])
def test_complete_cpu_path(algorithm):
    torch.set_num_threads(1)
    cfg = small(search=algorithm)
    task = task_spec(cfg)
    model = build_model(cfg, task)
    search = Search(model, cfg)
    games = play_games(search, cfg, [0, 1], 0)
    assert len(games) == 2
    replay = Replay(cfg.replay_games, task)
    for game in games:
        replay.add(game)
    batch = make_batch(replay, cfg, np.random.default_rng(8))
    learner = Learner(model, cfg, System(microbatch=2), "cpu")
    result = learner.step(batch)
    assert np.isfinite(result["loss"])
    assert result["learner_samples"] == cfg.batch_size


def test_halving_and_uneven_shards():
    np.testing.assert_array_equal(halving_visits(1, 5), np.arange(5))
    np.testing.assert_array_equal(halving_visits(16, 16), np.zeros(16))
    assert len(halving_visits(7, 50)) == 50
    slices = [shard_bounds(1024, rank, 6) for rank in range(6)]
    assert slices[0][0] == 0 and slices[-1][1] == 1024
    assert all(slices[i][1] == slices[i + 1][0] for i in range(5))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("algorithm", ["gumbel", "puct"])
def test_gpu_tree_matches_cpu(algorithm):
    torch.set_num_threads(1)
    torch.manual_seed(41)
    cfg = small(search=algorithm, simulations=12, considered_actions=4)
    env = make_environment(cfg, 0)
    positions = [env.position()]
    env.step(0)
    positions.append(env.position())
    torch.cuda.set_device(0)
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    model = build_model(cfg, env.task).to(device)
    noise = np.random.default_rng(42).gumbel(size=(2, env.task.actions)).astype(np.float32)
    cpu = Search(model, cfg, device, gpu_search=False)(
        positions, [np.random.default_rng(0) for _ in positions], explore=False, noise=noise
    )
    gpu = Search(model, cfg, device, gpu_search=True)(
        positions, [np.random.default_rng(0) for _ in positions], explore=False, noise=noise
    )
    np.testing.assert_array_equal(cpu.visits, gpu.visits)
    np.testing.assert_allclose(cpu.policy, gpu.policy, atol=2e-5)
    np.testing.assert_allclose(cpu.value, gpu.value, atol=2e-5)
    assert torch.cuda.current_device() == 0


def test_reference_weight_import():
    from pathlib import Path

    from zero_systems.reference import MiniZeroReference, read_weights

    path = Path(
        "/workspace/masters/Project-Data-Science/artifacts/baselines/minizero-selftrained/weight_iter_001200.pt"
    )
    if not path.exists():
        pytest.skip("reference artifact not installed")
    torch.set_num_threads(1)
    cfg = Experiment()
    model = build_model(cfg, task_spec(cfg)).eval()
    model.load_state_dict(read_weights(path))
    reference = MiniZeroReference(path)
    observations = torch.randint(2, (2, 18, 9, 9)).float()
    contexts = torch.empty(2, 0)
    with torch.inference_mode():
        actual = model.initial_inference(observations, contexts)
        expected = reference.initial_inference(observations, contexts)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        actions = torch.tensor([4, 81])
        actual = model.recurrent_inference(actual[0], actions)
        expected = reference.recurrent_inference(expected[0], actions)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
