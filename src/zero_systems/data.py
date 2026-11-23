"""Records exchanged by the environment, search, replay and learner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TaskSpec:
    observation_shape: tuple[int, ...]
    actions: int
    players: int
    context: int = 0
    encoding: str = "vector-v1"


@dataclass
class Position:
    observation: np.ndarray
    legal: np.ndarray
    player: int
    context: np.ndarray


@dataclass
class Trajectory:
    game_id: int
    seed: int
    observations: np.ndarray
    contexts: np.ndarray
    players: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    generating_update: int
    terminated: bool
    truncated: bool
    bootstrap: float = 0.0
    bootstrap_player: int = 0
    outcome: np.ndarray | None = None
    resigned_player: int = -1
    resignation_test_player: int = -1

    def __len__(self):
        return len(self.actions)

    def validate(self, task: TaskSpec) -> None:
        length = len(self)
        for player in (self.resigned_player, self.resignation_test_player):
            if not isinstance(player, (int, np.integer)) or not -1 <= player < task.players:
                raise ValueError("invalid resignation player")
        if self.resigned_player >= 0 and self.resignation_test_player >= 0:
            raise ValueError("a resigned game cannot test its counterfactual outcome")
        if max(self.resigned_player, self.resignation_test_player) >= 0 and self.outcome is None:
            raise ValueError("resignation records require an episodic outcome")
        if any(
            not isinstance(value, (int, np.integer)) or value < 0
            for value in (self.game_id, self.seed, self.generating_update)
        ):
            raise ValueError("invalid trajectory identity")
        if self.actions.shape != (length,) or not np.issubdtype(self.actions.dtype, np.integer):
            raise ValueError("actions must be a vector of integers")
        if not np.issubdtype(self.players.dtype, np.integer):
            raise ValueError("players must be integers")
        if (
            not isinstance(self.bootstrap_player, (int, np.integer))
            or not 0 <= self.bootstrap_player < task.players
        ):
            raise ValueError("invalid bootstrap player")
        if length == 0 or self.observations.shape != (length, *task.observation_shape):
            raise ValueError("invalid trajectory observations")
        expected = {
            "contexts": (length, task.context),
            "players": (length,),
            "rewards": (length, task.players),
            "policies": (length, task.actions),
            "values": (length,),
        }
        for key, shape in expected.items():
            value = getattr(self, key)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid trajectory {key}")
        if not np.isfinite(self.observations).all() or not np.isfinite(self.bootstrap):
            raise ValueError("non-finite trajectory")
        if ((self.actions < 0) | (self.actions >= task.actions)).any():
            raise ValueError("invalid trajectory action")
        if ((self.players < 0) | (self.players >= task.players)).any():
            raise ValueError("invalid trajectory player")
        if task.players == 2 and not np.allclose(self.rewards.sum(1), 0, atol=1e-6):
            raise ValueError("two-player rewards must be zero-sum")
        if (self.policies < 0).any() or not np.allclose(self.policies.sum(1), 1, atol=1e-5):
            raise ValueError("policies must be probability distributions")
        if self.terminated == self.truncated:
            raise ValueError("a finished trajectory must terminate or truncate")
        if self.outcome is not None:
            if not self.terminated or self.outcome.shape != (task.players,):
                raise ValueError("invalid episodic outcome")
            if not np.isfinite(self.outcome).all() or np.any(self.rewards):
                raise ValueError(
                    "outcome and transition rewards must not count the same return twice"
                )
            if task.players == 2 and not np.isclose(self.outcome.sum(), 0, atol=1e-6):
                raise ValueError("two-player outcomes must be zero-sum")


@dataclass
class Batch:
    observation: torch.Tensor
    context: torch.Tensor
    actions: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    valid: torch.Tensor
    reward_valid: torch.Tensor
    game_ids: torch.Tensor
    model_steps: torch.Tensor

    def to(self, device):
        return Batch(**{k: v.to(device, non_blocking=True) for k, v in vars(self).items()})

    def part(self, start: int, stop: int):
        return Batch(**{k: v[start:stop] for k, v in vars(self).items()})

    def __len__(self):
        return self.actions.shape[0]


def game_seed(run_seed: int, game_id: int, stream: int = 0) -> int:
    return int(np.random.SeedSequence([run_seed, game_id, stream]).generate_state(1)[0])
