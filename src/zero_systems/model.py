"""MuZero models are ordinary nn.Modules with named h/g/f components.

initial_inference(observation, context) returns (latent, policy_logits, value).
recurrent_inference(latent, action) returns (latent, policy_logits, value, reward).
Batch is the first axis; policy uses zero-based environment actions. Values and
rewards follow Experiment.value_perspective. Search treats latent axes as opaque.
The representation, dynamics and prediction attributes are the transfer units.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import Experiment
from .data import TaskSpec


def normalize_latent(x):
    flat = x.flatten(1)
    low = flat.amin(1).reshape(-1, *([1] * (x.ndim - 1)))
    high = flat.amax(1).reshape_as(low)
    return (x - low) / (high - low).clamp_min(1e-5)


def conv_norm(inputs, outputs, kernel):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel, padding=kernel // 2), nn.BatchNorm2d(outputs), nn.ReLU()
    )


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.first = conv_norm(channels, channels, 3)
        self.second = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(x + self.second(self.first(x)))


class ResidualModel(nn.Module):
    def __init__(self, config: Experiment, task: TaskSpec):
        super().__init__()
        if len(task.observation_shape) != 3:
            raise ValueError("residual model requires CHW observations")
        inputs, height, width = task.observation_shape
        if task.actions != height * width + 1:
            raise ValueError("this residual action encoder requires board points plus pass")
        self.task, self.config = task, config
        self.representation = nn.Sequential(
            conv_norm(inputs + task.context, config.channels, 3),
            *[Residual(config.channels) for _ in range(config.blocks)],
        )
        self.dynamics = nn.Sequential(
            conv_norm(config.channels + 1, config.channels, 3),
            *[Residual(config.channels) for _ in range(config.blocks)],
        )
        if config.prediction_head == "spatial":
            self.prediction = SpatialPrediction(config.channels, config.value_hidden)
            return
        self.prediction = nn.ModuleDict(
            {
                "policy": nn.Sequential(
                    conv_norm(config.channels, 2, 1),
                    nn.Flatten(),
                    nn.Linear(2 * height * width, task.actions),
                ),
                "value": nn.Sequential(
                    conv_norm(config.channels, 1, 1),
                    nn.Flatten(),
                    nn.Linear(height * width, config.value_hidden),
                    nn.ReLU(),
                    nn.Linear(config.value_hidden, 1),
                    nn.Tanh(),
                ),
            }
        )
        if config.value_bottleneck_slope:
            self.prediction["value"][0][2] = nn.LeakyReLU(config.value_bottleneck_slope)
        if not config.value_bottleneck_bias:
            # A negative offset can deactivate the entire single-channel ReLU.
            self.prediction["value"][0][1].register_parameter("bias", None)

    def initial_inference(self, observation, context):
        x = observation.float()
        if self.task.context:
            planes = context[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat((x, planes), 1)
        latent = normalize_latent(self.representation(x))
        if self.config.prediction_head == "spatial":
            policy, value = self.prediction(latent)
            return latent, policy, value
        return (
            latent,
            self.prediction["policy"](latent),
            self.prediction["value"](latent).squeeze(-1),
        )

    def recurrent_inference(self, latent, action):
        height, width = latent.shape[-2:]
        plane = F.one_hot(action.long(), height * width + 1)[:, :-1]
        plane = plane.reshape(-1, 1, height, width).to(latent.dtype)
        state = normalize_latent(self.dynamics(torch.cat((latent, plane), 1)))
        if self.config.prediction_head == "spatial":
            policy, value = self.prediction(state)
            return state, policy, value, torch.zeros_like(value)
        policy = self.prediction["policy"](state)
        value = self.prediction["value"](state).squeeze(-1)
        return state, policy, value, torch.zeros_like(value)


class SpatialPrediction(nn.Module):
    """Shared point logits conditioned on pooled state; pass is the last action.

    Mean pooling keeps feature scale stable across sizes. Explicit height/width
    features distinguish boards with otherwise similar pooled states. No learned
    tensor has a board-sized axis.
    """

    def __init__(self, channels, hidden):
        super().__init__()
        self.points = nn.Conv2d(channels, channels, 1)
        self.context = nn.Linear(channels + 2, channels)
        self.logit = nn.Conv2d(channels, 1, 1)
        self.pass_move = nn.Sequential(
            nn.Linear(channels + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.value = nn.Sequential(
            nn.Linear(channels + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Tanh()
        )

    def forward(self, state):
        height, width = state.shape[-2:]
        size = state.new_ones((state.shape[0], 2)) * state.new_tensor([height / 19, width / 19])
        pooled = torch.cat((state.mean((-2, -1)), size), 1)
        points = F.relu(self.points(state) + self.context(pooled)[:, :, None, None])
        policy = torch.cat((self.logit(points).flatten(1), self.pass_move(pooled)), 1)
        return policy, self.value(pooled).squeeze(-1)


class BoardAttention(nn.Module):
    """Point tokens with fixed 2D sinusoidal coordinates in [0, 1]."""

    def __init__(self, inputs, config):
        super().__init__()
        self.input = nn.Conv2d(inputs, config.channels, 1)
        layer = nn.TransformerEncoderLayer(
            config.channels,
            config.attention_heads,
            4 * config.channels,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(
            layer, config.blocks, norm=nn.LayerNorm(config.channels), enable_nested_tensor=False
        )
        # TransformerEncoder clones one layer; initialize each block independently.
        for block in self.layers.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            for linear in (block.self_attn.out_proj, block.linear1, block.linear2):
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)

    def forward(self, board):
        state = self.input(board)
        batch, channels, height, width = state.shape
        frequencies = (
            2
            * math.pi
            * 2.0 ** torch.arange(channels // 4, device=state.device).float().div(channels // 4)
        )
        rows = torch.linspace(0, 1, height, device=state.device)[:, None] * frequencies
        cols = torch.linspace(0, 1, width, device=state.device)[:, None] * frequencies
        row = torch.cat((rows.sin(), rows.cos()), -1)[:, None, :].expand(height, width, -1)
        col = torch.cat((cols.sin(), cols.cos()), -1)[None, :, :].expand(height, width, -1)
        position = torch.cat((row, col), -1).reshape(1, height * width, channels).to(state.dtype)
        tokens = state.flatten(2).transpose(1, 2) + position
        return self.layers(tokens).transpose(1, 2).reshape(batch, channels, height, width)


class TransformerModel(nn.Module):
    def __init__(self, config: Experiment, task: TaskSpec):
        super().__init__()
        if (
            len(task.observation_shape) != 3
            or task.actions != math.prod(task.observation_shape[-2:]) + 1
        ):
            raise ValueError("transformer requires CHW board observations and points plus pass")
        self.task, self.config = task, config
        self.representation = BoardAttention(task.observation_shape[0] + task.context, config)
        self.dynamics = BoardAttention(config.channels + 2, config)
        self.prediction = SpatialPrediction(config.channels, config.value_hidden)

    def initial_inference(self, observation, context):
        board = observation.float()
        if self.task.context:
            board = torch.cat(
                (board, context[:, :, None, None].expand(-1, -1, *board.shape[-2:])), 1
            )
        state = normalize_latent(self.representation(board))
        policy, value = self.prediction(state)
        return state, policy, value

    def recurrent_inference(self, latent, action):
        height, width = latent.shape[-2:]
        encoded = F.one_hot(action.long(), height * width + 1).to(latent.dtype)
        point = encoded[:, :-1].reshape(-1, 1, height, width)
        passed = encoded[:, -1, None, None, None].expand(-1, 1, height, width)
        state = normalize_latent(self.dynamics(torch.cat((latent, point, passed), 1)))
        policy, value = self.prediction(state)
        return state, policy, value, torch.zeros_like(value)


class MLPModel(nn.Module):
    def __init__(self, config: Experiment, task: TaskSpec):
        super().__init__()
        self.task, self.config = task, config
        size = config.channels
        self.representation = nn.Sequential(
            nn.Linear(math.prod(task.observation_shape) + task.context, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(size + task.actions, size), nn.ReLU(), nn.Linear(size, size), nn.ReLU()
        )
        self.prediction = nn.ModuleDict(
            {
                "policy": nn.Linear(size, task.actions),
                "value": nn.Linear(size, 1),
                "reward": nn.Linear(size, 1),
            }
        )

    def initial_inference(self, observation, context):
        latent = normalize_latent(
            self.representation(torch.cat((observation.flatten(1).float(), context), 1))
        )
        return (
            latent,
            self.prediction["policy"](latent),
            self.prediction["value"](latent).squeeze(-1),
        )

    def recurrent_inference(self, latent, action):
        encoding = F.one_hot(action.long(), self.task.actions).to(latent.dtype)
        state = normalize_latent(self.dynamics(torch.cat((latent, encoding), 1)))
        return (
            state,
            self.prediction["policy"](state),
            self.prediction["value"](state).squeeze(-1),
            self.prediction["reward"](state).squeeze(-1),
        )


def build_model(config: Experiment, task: TaskSpec):
    return {"residual": ResidualModel, "mlp": MLPModel, "transformer": TransformerModel}[
        config.network
    ](config, task)


def initialize(
    model: nn.Module,
    source: dict,
    components=("representation", "dynamics", "prediction"),
    *,
    source_task=None,
    source_config=None,
):
    """Transfer named components, requiring matching meanings and tensor shapes."""
    if (
        not components
        or len(set(components)) != len(components)
        or not set(components) <= {"representation", "dynamics", "prediction"}
    ):
        raise ValueError("unknown transfer component")
    strict = model.config.prediction_head == "spatial"
    if strict:
        if source_task is None or source_config is None:
            raise ValueError("spatial transfer requires checkpoint task and architecture metadata")
        keys = [
            "network",
            "prediction_head",
            "channels",
            "blocks",
            "value_hidden",
            "history",
            "condition_komi",
            "value_perspective",
            "discount",
            "environment",
            "environment_kwargs",
        ]
        if model.config.network == "transformer":
            keys.append("attention_heads")
        if any(getattr(source_config, k) != getattr(model.config, k) for k in keys):
            raise ValueError("initialization changes architecture, encoding or value semantics")
        if source_task.context != model.task.context:
            raise ValueError("initialization changes context encoding")
    if source_task is not None:
        if source_task.encoding != model.task.encoding or source_task.players != model.task.players:
            raise ValueError("initialization changes the meaning of observations or values")
        if (
            source_task.encoding.startswith("go-")
            and source_task.observation_shape[0] != model.task.observation_shape[0]
        ):
            raise ValueError("initialization changes history encoding")
    target = model.state_dict()
    if strict:

        def selected(weights):
            return {k for k in weights if k.split(".")[0] in components}

        if selected(source) != selected(target):
            raise ValueError("selected component tensors do not match")
    copied, reset, expanded = [], [], []
    for key, value in target.items():
        if key.split(".")[0] not in components:
            reset.append(key)
        elif key in source and source[key].shape == value.shape:
            target[key] = source[key]
            copied.append(key)
        elif strict:
            raise ValueError(f"incompatible initialization tensor: {key}")
        elif (
            key == "representation.0.0.weight"
            and source_task is not None
            and source_task.context == 0
            and model.task.context == 1
            and key in source
            and source[key].shape == (value.shape[0], value.shape[1] - 1, *value.shape[2:])
        ):
            target[key] = torch.zeros_like(value)
            target[key][:, :-1] = source[key]
            expanded.append(key)
        elif key.startswith(("prediction.policy.2.", "prediction.value.2.")):
            reset.append(key)
        else:
            raise ValueError(f"incompatible initialization tensor: {key}")
    model.load_state_dict(target)
    return {"copied": copied, "reset": reset, "expanded": expanded}


def initialize_checkpoint(model, path, components=("representation", "dynamics", "prediction")):
    """Copy weights only and return portable provenance, never source training state."""
    import hashlib
    from pathlib import Path

    from .config import from_dict
    from .reference import read_weights

    weights, task, perspective, record = read_weights(path, with_record=True)
    if perspective is not None and perspective != model.config.value_perspective:
        raise ValueError("initialization changes value perspective")
    config = from_dict(record["experiment"], Experiment) if "experiment" in record else None
    report = initialize(model, weights, components, source_task=task, source_config=config)
    with Path(path).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "source": str(Path(path).resolve()),
        "source_sha256": digest,
        "source_progress": record.get("progress"),
        "source_experiment": record.get("experiment"),
        "source_task": record.get("task"),
        "components": list(components),
        "source_initialization": record.get("initialization"),
        "fresh": ["optimizer", "replay", "rng", "stage_counters"],
        "fine_tune": "all_parameters",
        **report,
    }
