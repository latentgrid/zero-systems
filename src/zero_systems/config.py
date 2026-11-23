"""Experiment choices and machine choices are deliberately separate."""

from __future__ import annotations

import dataclasses
import json
import math
import tomllib
from fractions import Fraction
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin, get_type_hints


@dataclasses.dataclass(frozen=True)
class Experiment:
    environment: str = "go"
    environment_kwargs: dict = dataclasses.field(default_factory=dict)
    action_mask_key: str = "action_mask"
    action_mask_source: str = "info"
    board_size: int = 9
    history: int = 8
    komi: tuple[float, ...] = (7.0,)
    condition_komi: bool = False
    max_moves: int = 163
    network: str = "residual"
    prediction_head: str = "dense"
    attention_heads: int = 4
    channels: int = 256
    blocks: int = 3
    value_hidden: int = 256
    value_bottleneck_slope: float = 0.0
    value_bottleneck_bias: bool = True
    value_perspective: str = "player0"
    search: str = "gumbel"
    simulations: int = 16
    considered_actions: int = 16
    discount: float = 1.0
    puct_base: float = 19652.0
    puct_init: float = 1.25
    dirichlet_alpha: float = 0.03
    dirichlet_fraction: float = 0.25
    rescale_q: bool = False
    gumbel_scale: float = 1.0
    q_scale: float = 1.0
    action_threshold: float = 0.1
    temperature: float = 1.0
    temperature_moves: int = 163
    resign_threshold: float = -0.9
    resign_after: int = 1200
    no_resign_fraction: float = 0.1
    batch_size: int = 1024
    unroll: int = 5
    td_steps: int = 163
    replay_games: int = 40000
    augmentation: bool = True
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0001
    grad_clip: float = 5.0
    value_weight: float = 1.0
    reward_weight: float = 0.0
    schedule: str = "phased"
    games_per_round: int = 2000
    updates_per_round: int = 200
    warmup_games: int = 2000
    game_lead: int = 2000
    games_per_update: float | None = None
    samples_per_position: float | None = None
    warmup_positions: int = 10000
    position_lead: int = 20000
    max_positions: int | None = None
    publish_updates: int = 200
    seed: int = 303
    updates: int = 12000
    initialize: str | None = None

    @property
    def game_ratio(self):
        return (
            Fraction(str(self.games_per_update))
            if self.games_per_update is not None
            else Fraction(self.games_per_round, self.updates_per_round)
        )

    @property
    def position_ratio(self):
        return Fraction(str(self.samples_per_position)) if self.samples_per_position else None

    def can_update(self, games: int, positions: int, samples: int, updates: int):
        if self.position_ratio is not None:
            return positions >= self.warmup_positions and samples + self.batch_size <= (
                self.position_ratio * positions
            )
        return games >= self.warmup_games and (updates + 1) * self.game_ratio <= games

    def validate(self) -> None:
        if not isinstance(self.environment_kwargs, dict):
            raise ValueError("environment_kwargs must be a table")
        json.dumps(self.environment_kwargs, allow_nan=False)
        if self.action_mask_source not in {"info", "observation"} or not self.action_mask_key:
            raise ValueError("action mask requires an info/observation source and a key")
        if self.games_per_update is not None and self.samples_per_position is not None:
            raise ValueError("choose exactly one asynchronous ratio")
        for name in ("games_per_update", "samples_per_position"):
            value = getattr(self, name)
            if value is not None and (self.schedule != "async" or value <= 0):
                raise ValueError(f"{name} requires async scheduling and a positive ratio")
        if self.samples_per_position is not None:
            if self.warmup_positions <= 0 or self.position_lead < max(
                self.warmup_positions, math.ceil(self.batch_size / self.position_ratio)
            ):
                raise ValueError("position lead must permit warmup and a complete learner batch")
        if self.max_positions is not None:
            if self.position_ratio is None or type(self.max_positions) is not int:
                raise ValueError("max_positions requires async position scheduling and an integer")
            if self.max_positions < max(
                self.warmup_positions, math.ceil(self.batch_size / self.position_ratio)
            ):
                raise ValueError("position budget must permit warmup and a complete learner batch")
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        for key in (
            "board_size",
            "history",
            "max_moves",
            "channels",
            "blocks",
            "value_hidden",
            "simulations",
            "considered_actions",
            "batch_size",
            "unroll",
            "td_steps",
            "replay_games",
            "games_per_round",
            "updates_per_round",
            "warmup_games",
            "game_lead",
            "publish_updates",
            "updates",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.search not in {"puct", "gumbel"}:
            raise ValueError("search must be puct or gumbel")
        if self.network not in {"residual", "mlp", "transformer"}:
            raise ValueError("network must be residual, mlp or transformer")
        if self.prediction_head not in {"dense", "spatial"}:
            raise ValueError("prediction_head must be dense or spatial")
        if self.network == "mlp" and self.prediction_head != "dense":
            raise ValueError("spatial heads require a board network")
        if self.network == "transformer" and (
            self.prediction_head != "spatial"
            or self.attention_heads <= 0
            or self.channels % self.attention_heads
            or self.channels % 4
        ):
            raise ValueError(
                "transformer requires spatial heads and channels divisible by heads and 4"
            )
        if self.prediction_head == "spatial" and (
            self.value_bottleneck_slope != 0 or not self.value_bottleneck_bias
        ):
            raise ValueError("value bottleneck options apply only to dense residual heads")
        if not 0 <= self.value_bottleneck_slope <= 1 or (
            self.network != "residual" and self.value_bottleneck_slope != 0
        ):
            raise ValueError(
                "value bottleneck slope requires a residual network and lies in [0, 1]"
            )
        if self.value_perspective not in {"player0", "to_play"}:
            raise ValueError("unknown value perspective")
        if self.network != "residual" and not self.value_bottleneck_bias:
            raise ValueError("a fixed value bottleneck offset requires a residual network")
        if self.schedule not in {"phased", "async"}:
            raise ValueError("schedule must be phased or async")
        if not self.komi or not all(math.isfinite(k) for k in self.komi):
            raise ValueError("komi must contain finite values")
        if len(self.komi) > 1 and not self.condition_komi:
            raise ValueError("variable komi requires conditioning")
        if not 0 <= self.discount <= 1 or self.learning_rate <= 0:
            raise ValueError("invalid discount or learning rate")
        if not 0 <= self.no_resign_fraction <= 1 or not 0 <= self.dirichlet_fraction <= 1:
            raise ValueError("fractions must lie in [0, 1]")
        if self.temperature <= 0 or self.dirichlet_alpha <= 0 or self.puct_base <= 0:
            raise ValueError("temperature, alpha and PUCT base must be positive")
        if not 0 <= self.momentum < 1 or self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if (
            min(
                self.value_weight,
                self.reward_weight,
                self.q_scale,
                self.gumbel_scale,
                self.action_threshold,
                self.puct_init,
                self.seed,
                self.temperature_moves,
                self.resign_after,
            )
            < 0
        ):
            raise ValueError("loss weights, search scales and counters must be nonnegative")
        if self.samples_per_position is None and self.warmup_games > self.replay_games:
            raise ValueError("replay cannot hold the warmup")
        if self.schedule == "phased" and self.warmup_games > self.games_per_round:
            raise ValueError("the first phased collection must cover warmup_games")
        if (
            self.schedule == "async"
            and self.samples_per_position is None
            and self.game_lead < max(self.warmup_games, math.ceil(self.game_ratio))
        ):
            raise ValueError("actor lead must permit warmup to finish")
        if (
            self.schedule == "async"
            and self.samples_per_position is None
            and self.updates * self.game_ratio < self.warmup_games
        ):
            raise ValueError("update budget does not permit asynchronous warmup")
        if (
            self.schedule == "async"
            and self.samples_per_position is None
            and (self.updates * self.game_ratio).denominator != 1
        ):
            raise ValueError("game ratio and update budget must specify a whole number of games")
        if self.environment != "go" and self.augmentation:
            raise ValueError("dihedral augmentation requires the Go action layout")
        if self.environment == "go" and (not 2 <= self.board_size <= 19 or self.max_moves < 2):
            raise ValueError("Go requires board size 2..19 and a move limit of at least two")
        if self.environment == "go" and set(self.environment_kwargs) & {
            "board_size",
            "komi",
            "max_game_length",
        }:
            raise ValueError("use the explicit Go board_size, komi and max_moves settings")


def _validate_device(name, value, precision, *, explicit_cuda=False):
    import torch

    try:
        device = torch.device(value)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"{name}: invalid device {value!r}") from exc
    index = value.partition(":")[2]
    if index and device.index != int(index):
        raise ValueError(f"{name}: device {value!r} resolves to {str(device)!r}")
    if explicit_cuda and device.type == "cuda" and device.index is None:
        raise ValueError(f"{name}: use an explicit CUDA index such as cuda:0")
    if precision == "bfloat16" and device.type != "cuda":
        raise ValueError(f"{name}: bfloat16 precision requires CUDA")


@dataclasses.dataclass(frozen=True)
class Evaluation:
    name: str
    mode: str = "fixed"
    interval_updates: int = 6000
    games: int = 100
    seed: int = 7301
    simulations: int = 16
    search: str | None = None
    opening_moves: int = 2
    opponent: str = "random"
    lag_updates: int = 6000
    device: str = "cpu"
    batch: int = 32
    precision: str = "float32"

    def validate(self):
        if not self.name or not all(c.isalnum() or c in "-_" for c in self.name):
            raise ValueError("evaluation name must contain only letters, digits, - or _")
        if self.mode not in {"fixed", "episodes", "self_lag"}:
            raise ValueError("evaluation mode must be fixed, episodes or self_lag")
        if (
            min(self.interval_updates, self.games, self.simulations, self.batch, self.lag_updates)
            <= 0
        ):
            raise ValueError("evaluation counts and intervals must be positive")
        if self.mode != "episodes" and (self.games % 2 or self.batch < 2):
            raise ValueError("competitive evaluation requires even games and batch >= 2")
        if self.seed < 0 or self.opening_moves < 0 or self.search not in {None, "puct", "gumbel"}:
            raise ValueError("invalid evaluation seed, openings or search")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("invalid evaluation precision")
        _validate_device("evaluation device", self.device, self.precision)


def accumulation_passes(batch_size: int, learners: int, microbatch: int) -> int:
    """All DDP ranks must make the same number of nonempty backward passes."""
    smallest, remainder = divmod(batch_size, learners)
    largest = smallest + bool(remainder)
    passes = (largest + microbatch - 1) // microbatch
    if smallest < passes or smallest == 0:
        raise ValueError(
            f"batch_size={batch_size} across {learners} learners with microbatch={microbatch} "
            "cannot give every learner a sample on every pass"
        )
    return passes


@dataclasses.dataclass(frozen=True)
class System:
    learner_devices: tuple[str, ...] = ("cpu",)
    actor_devices: tuple[str, ...] = ()
    actors_per_device: int = 1
    actor_batch: int = 64
    microbatch: int = 256
    precision: str = "float32"
    actor_precision: str = "float32"
    compile: bool = False
    gpu_search: bool = True
    threads: int = 1
    deterministic: bool = False
    ddp_bucket_mb: int = 5
    queue_batches: int = 8
    timeout: float = 300.0
    checkpoint_updates: int = 10000
    model_updates: int = 200
    log_updates: int = 25
    max_seconds: float = 0.0
    listen: str | None = None
    auth_environment: str = "ZERO_SYSTEMS_AUTHKEY"
    tensorboard: bool = False
    evaluations: tuple[Evaluation, ...] = ()

    def validate(self, batch_size: int | None = None) -> None:
        for evaluation in self.evaluations:
            evaluation.validate()
        if len({e.name for e in self.evaluations}) != len(self.evaluations):
            raise ValueError("evaluation names must be unique")
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        for key in (
            "actors_per_device",
            "actor_batch",
            "microbatch",
            "threads",
            "queue_batches",
            "checkpoint_updates",
            "model_updates",
            "log_updates",
            "ddp_bucket_mb",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not self.learner_devices or self.timeout <= 0 or self.max_seconds < 0:
            raise ValueError("invalid devices or timeout")
        for device in self.learner_devices:
            _validate_device("learner_devices", device, self.precision, explicit_cuda=True)
        actor_devices = self.actor_devices or (() if self.listen else self.learner_devices[:1])
        for device in actor_devices:
            _validate_device("actor_devices", device, self.actor_precision, explicit_cuda=True)
        if len(set(self.learner_devices)) != len(self.learner_devices):
            raise ValueError("learner devices must be distinct")
        if len(self.learner_devices) > 1 and not all(
            d.startswith("cuda") for d in self.learner_devices
        ):
            raise ValueError("distributed learning requires CUDA")
        for name in ("precision", "actor_precision"):
            if getattr(self, name) not in {"float32", "bfloat16"}:
                raise ValueError("supported precisions: float32 and bfloat16")
        if batch_size is not None:
            accumulation_passes(batch_size, len(self.learner_devices), self.microbatch)


def load(path: str | Path, kind: type[Experiment] | type[System]):
    path = Path(path)
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    result = from_dict(values, kind)
    if isinstance(result, Experiment) and result.initialize:
        source = Path(result.initialize)
        if not source.is_absolute():
            result = dataclasses.replace(result, initialize=str((path.parent / source).resolve()))
    return result


def _setting(value, annotation, name):
    if get_origin(annotation) is UnionType:
        # Optional settings accept None in saved checkpoints.
        if value is None:
            return None
        annotation = next(t for t in get_args(annotation) if t is not type(None))
    if get_origin(annotation) is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{name} must be an array")
        item_type = get_args(annotation)[0]
        return tuple(_setting(item, item_type, f"{name}[{i}]") for i, item in enumerate(value))
    if annotation is Evaluation and isinstance(value, dict):
        if "name" not in value:
            raise ValueError(f"{name}.name is required")
        return from_dict(value, Evaluation)
    if type(value) is annotation or (annotation is float and type(value) is int):
        return value
    raise ValueError(f"{name} must be {annotation.__name__}")


def from_dict(values: dict, kind: type[Experiment] | type[System] | type[Evaluation]):
    types = get_type_hints(kind)
    if unknown := set(values) - types.keys():
        raise ValueError(f"unknown {kind.__name__} settings: {sorted(unknown)}")
    result = kind(**{k: _setting(v, types[k], k) for k, v in values.items()})
    result.validate()
    return result


def learning_settings(config: Experiment):
    """Canonical settings for resume, including legacy game-ratio configurations."""
    result = dataclasses.asdict(config)
    for key in ("updates", "initialize", "max_positions"):
        result.pop(key)
    if config.schedule == "async":
        for key in ("games_per_round", "updates_per_round", "games_per_update"):
            result.pop(key)
        if config.samples_per_position is None:
            result["game_ratio"] = str(config.game_ratio)
    return result


def identity(value) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(dataclasses.asdict(value), sort_keys=True).encode()
    ).hexdigest()
