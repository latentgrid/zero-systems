"""Independently implemented representation, dynamics and prediction networks."""

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
        self.task = task
        self.representation = nn.Sequential(
            conv_norm(inputs + task.context, config.channels, 3),
            *[Residual(config.channels) for _ in range(config.blocks)],
        )
        self.dynamics = nn.Sequential(
            conv_norm(config.channels + 1, config.channels, 3),
            *[Residual(config.channels) for _ in range(config.blocks)],
        )
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
        policy = self.prediction["policy"](state)
        value = self.prediction["value"](state).squeeze(-1)
        return state, policy, value, torch.zeros_like(value)


class MLPModel(nn.Module):
    def __init__(self, config: Experiment, task: TaskSpec):
        super().__init__()
        self.task = task
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
    return ResidualModel(config, task) if config.network == "residual" else MLPModel(config, task)


def initialize(
    model: nn.Module,
    source: dict,
    components=("representation", "dynamics", "prediction"),
    *,
    source_task=None,
):
    """Transfer named components, requiring matching meanings and tensor shapes."""
    if not set(components) <= {"representation", "dynamics", "prediction"}:
        raise ValueError("unknown transfer component")
    if source_task is not None:
        if source_task.encoding != model.task.encoding or source_task.players != model.task.players:
            raise ValueError("initialization changes the meaning of observations or values")
        if (
            source_task.encoding.startswith("go-")
            and source_task.observation_shape[0] != model.task.observation_shape[0]
        ):
            raise ValueError("initialization changes history encoding")
    target = model.state_dict()
    copied, reset, expanded = [], [], []
    for key, value in target.items():
        if key.split(".")[0] not in components:
            reset.append(key)
        elif key in source and source[key].shape == value.shape:
            target[key] = source[key]
            copied.append(key)
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
