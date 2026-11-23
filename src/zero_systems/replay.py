"""Uniform position sampling and explicit unrolled targets."""

from __future__ import annotations

from collections import OrderedDict, deque
from functools import lru_cache

import numba
import numpy as np
import torch

from .config import Experiment
from .data import Batch, TaskSpec, Trajectory


class Replay:
    def __init__(self, capacity: int, task: TaskSpec):
        self.capacity = capacity
        self.task = task
        self.games: OrderedDict[int, Trajectory] = OrderedDict()
        self._lengths = deque(maxlen=capacity)
        self.positions = 0
        self._index: list[Trajectory] = []
        self._ends = np.empty(0, np.int64)
        self._dirty = True
        self._targets = {}
        self._target_config = None

    def add(self, game: Trajectory) -> bool:
        game.validate(self.task)
        if game.game_id in self.games:
            return False
        self.games[game.game_id] = game
        self._lengths.append(len(game))
        self.positions += len(game)
        while len(self.games) > self.capacity:
            _, old = self.games.popitem(last=False)
            self.positions -= len(old)
            self._targets.pop(old.game_id, None)
        self._dirty = True
        return True

    def sample(self, count: int, rng: np.random.Generator):
        if self.positions == 0:
            raise ValueError("cannot sample empty replay")
        if self._dirty:
            self._index = list(self.games.values())
            self._ends = np.cumsum(self._lengths)
            self._dirty = False
        positions = rng.integers(self.positions, size=count)
        indices = np.searchsorted(self._ends, positions, side="right")
        starts = np.r_[0, self._ends[:-1]]
        return [
            (self._index[index], int(position - starts[index]))
            for index, position in zip(indices, positions, strict=True)
        ]


def value_targets(game: Trajectory, config: Experiment) -> np.ndarray:
    """Rewards at t belong to action a_t; a terminal outcome is a separate payoff."""
    tail = game.outcome if game.outcome is not None else np.zeros(game.rewards.shape[1], np.float32)
    return _returns(
        game.rewards,
        game.players,
        game.values,
        config.discount,
        config.td_steps,
        config.value_perspective == "player0",
        game.truncated,
        game.bootstrap,
        game.bootstrap_player,
        tail,
    )


@numba.njit(cache=True, nogil=True)
def _returns(
    rewards,
    players,
    values,
    discount,
    horizon,
    player0,
    truncated,
    bootstrap,
    bootstrap_player,
    outcome,
):
    length = len(players)
    # A discounted suffix makes all n-step targets linear in trajectory length.
    suffix = np.zeros((length + 1, rewards.shape[1]), np.float64)
    for t in range(length - 1, -1, -1):
        suffix[t] = rewards[t] + discount * suffix[t + 1]
    result = np.zeros(length, np.float32)
    for t in range(length):
        player = 0 if player0 else int(players[t])
        end = min(length, t + horizon)
        factor = discount ** (end - t)
        result[t] = suffix[t, player] - factor * suffix[end, player]
        if end < length:
            source_player = 0 if player0 else int(players[end])
            sign = 1 if player == source_player else -1
            result[t] += factor * sign * values[end]
        elif truncated:
            sign = 1 if player == bootstrap_player else -1
            result[t] += factor * sign * bootstrap
        else:
            result[t] += factor * outcome[player]
    return result


@lru_cache(maxsize=152)
def _symmetry_maps(side: int, transform: int):
    board = np.arange(side * side).reshape(side, side)
    if transform >= 4:
        board = np.fliplr(board)
    permutation = np.rot90(board, transform % 4).ravel()
    full = np.append(permutation, side * side)
    return permutation, full, np.argsort(full)


def make_batch(replay: Replay, config: Experiment, rng: np.random.Generator, samples=None) -> Batch:
    samples = replay.sample(config.batch_size, rng) if samples is None else samples
    batch, steps, task = len(samples), config.unroll + 1, replay.task
    observation = np.stack([game.observations[t] for game, t in samples])
    context = np.stack([game.contexts[t] for game, t in samples])
    actions = rng.integers(task.actions, size=(batch, steps - 1), dtype=np.int64)
    policy = np.zeros((batch, steps, task.actions), np.float32)
    values = np.zeros((batch, steps), np.float32)
    rewards = np.zeros_like(values)
    valid = np.zeros_like(values)
    reward_valid = np.zeros_like(values)
    target_config = (config.discount, config.td_steps, config.value_perspective)
    if replay._target_config != target_config:
        replay._targets.clear()
        replay._target_config = target_config
    # Accepted trajectories are immutable. Cache targets until their game is evicted.
    cache = replay._targets
    for row, (game, t) in enumerate(samples):
        length = len(game)
        if game.game_id not in cache:
            cache[game.game_id] = value_targets(game, config)
        targets = cache[game.game_id]
        live = min(steps, length - t)
        policy[row, :live] = game.policies[t : t + live]
        values[row, :live] = targets[t : t + live]
        valid[row, :live] = 1
        if live < steps and game.outcome is not None:
            policy[row, live:].fill(1 / task.actions)
            if config.value_perspective == "player0":
                values[row, live:] = game.outcome[0]
            else:
                players = (game.players[-1] + np.arange(1, steps - live + 1)) % task.players
                values[row, live:] = game.outcome[players]
            valid[row, live:] = 1
        transitions = min(steps - 1, length - t)
        actions[row, :transitions] = game.actions[t : t + transitions]
        if config.value_perspective == "player0":
            rewards[row, 1 : transitions + 1] = game.rewards[t : t + transitions, 0]
        else:
            states = np.arange(t, t + transitions)
            rewards[row, 1 : transitions + 1] = game.rewards[states, game.players[states]]
        reward_valid[row, 1 : transitions + 1] = 1
        if game.terminated and game.outcome is None:
            # Transition-reward games have no return left after the final reward.
            # Train the latent tail as absorbing; truncations still bootstrap.
            policy[row, live:].fill(1 / task.actions)
            valid[row, live:] = 1
            reward_valid[row, 1:] = 1
    if config.augmentation:
        side = task.observation_shape[-1]
        transforms = rng.integers(8, size=batch)
        permutations, full, inverse = map(
            np.stack, zip(*[_symmetry_maps(side, i) for i in range(8)], strict=True)
        )
        observation = np.take_along_axis(
            observation.reshape(batch, -1, side * side),
            permutations[transforms, None, :],
            axis=2,
        ).reshape(batch, *task.observation_shape)
        policy = np.take_along_axis(policy, full[transforms, None, :], axis=2)
        actions = inverse[transforms[:, None], actions]
    return Batch(
        *[
            torch.from_numpy(np.ascontiguousarray(value))
            for value in (
                observation,
                context,
                actions,
                policy,
                values,
                rewards,
                valid,
                reward_valid,
                np.array([g.game_id for g, _ in samples], np.int64),
                np.array([g.generating_update for g, _ in samples], np.int64),
            )
        ]
    )
