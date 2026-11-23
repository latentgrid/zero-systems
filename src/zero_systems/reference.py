"""Import reference *weights*, without importing another implementation's code."""

from __future__ import annotations

import re

import torch
from torch import nn

from .data import TaskSpec
from .storage import load_checkpoint


def convert_minizero_weights(weights):
    result = {}
    stems = {
        "representation_conv": "representation.0.0",
        "representation_bn": "representation.0.1",
        "dynamics_conv": "dynamics.0.0",
        "dynamics_bn": "dynamics.0.1",
        "policy_conv": "prediction.policy.0.0",
        "policy_bn": "prediction.policy.0.1",
        "policy_fc": "prediction.policy.2",
        "value_conv": "prediction.value.0.0",
        "value_bn": "prediction.value.0.1",
        "value_fc1": "prediction.value.2",
        "value_fc2": "prediction.value.4",
    }
    for key, tensor in weights.items():
        for component in ("representation", "dynamics"):
            key = key.replace(f"{component}_network.residual_blocks.", f"{component}_blocks.")
            key = key.replace(f"{component}_network.conv.", f"{component}_conv.")
            key = key.replace(f"{component}_network.bn.", f"{component}_bn.")
        key = key.replace("prediction_network.policy.", "policy_")
        key = key.replace("prediction_network.value.", "value_")
        prefix, suffix = key.split(".", 1)
        if prefix in stems:
            result[f"{stems[prefix]}.{suffix}"] = tensor
        else:
            match = re.fullmatch(
                r"(representation|dynamics)_blocks\.(\d+)\.(conv1|conv2|bn1|bn2)\.(.+)", key
            )
            if not match:
                raise ValueError(f"unsupported reference tensor {key}")
            component, block, layer, attribute = match.groups()
            layer = {"conv1": "first.0", "bn1": "first.1", "conv2": "second.0", "bn2": "second.1"}[
                layer
            ]
            result[f"{component}.{int(block) + 1}.{layer}.{attribute}"] = tensor
    return result


def read_weights(path, *, with_metadata=False):
    task, perspective = None, None
    try:
        script = torch.jit.load(str(path), map_location="cpu")
    except (RuntimeError, ValueError):
        # Initialization/checkpoints are trusted local research artifacts.
        record = load_checkpoint(path)
        weights = record.get("model", record)
        if record.get("kind") in {"model", "full"}:
            task = TaskSpec(
                **{
                    **record["task"],
                    "observation_shape": tuple(record["task"]["observation_shape"]),
                }
            )
            perspective = record["experiment"]["value_perspective"]
    else:
        weights = script.state_dict()
    reference_names = ("representation_conv.weight", "representation_network.conv.weight")
    if any(k in weights for k in reference_names):
        weights = convert_minizero_weights(weights)
        inputs = weights["representation.0.0.weight"].shape[1]
        side = int((weights["prediction.policy.2.bias"].shape[0] - 1) ** 0.5)
        task = TaskSpec((inputs, side, side), side * side + 1, 2, encoding="go-relative-history-v1")
        perspective = "player0"
    return (weights, task, perspective) if with_metadata else weights


def convert_checkpoint(source, config, output, updates):
    """Convert weight data only; unknown training counters remain unknown."""
    import dataclasses
    from pathlib import Path

    from .env import task_spec
    from .evaluate import fingerprint
    from .model import build_model
    from .storage import atomic_save

    if updates < 0 or Path(output).exists():
        raise ValueError("provide a nonnegative reference update and a new output path")
    if config.value_bottleneck_slope != 0 or not config.value_bottleneck_bias:
        raise ValueError(
            "reference conversion preserves the value bottleneck; use initialization to change it"
        )
    task = task_spec(config)
    model = build_model(config, task)
    weights, source_task, perspective = read_weights(source, with_metadata=True)
    if source_task is not None and source_task != task:
        raise ValueError("conversion must preserve the original task encoding")
    if perspective is not None and perspective != config.value_perspective:
        raise ValueError("conversion must preserve value perspective")
    model.load_state_dict(weights)
    atomic_save(
        Path(output),
        {
            "kind": "model",
            "format": 1,
            "model": model.state_dict(),
            "experiment": dataclasses.asdict(config),
            "task": dataclasses.asdict(task),
            "progress": {
                "updates": updates,
                "learner_samples": None,
                "games": None,
                "positions": None,
                "seconds": None,
            },
            "external_weights": fingerprint(source),
        },
    )


class MiniZeroReference(nn.Module):
    def __init__(self, path, device="cpu"):
        super().__init__()
        self.network = torch.jit.load(str(path), map_location=device).eval()
        net = self.network
        self.task = TaskSpec(
            (
                net.get_num_input_channels(),
                net.get_input_channel_height(),
                net.get_input_channel_width(),
            ),
            net.get_action_size(),
            2,
            encoding="go-relative-history-v1",
        )

    def initial_inference(self, observation, context):
        result = self.network.initial_inference(observation.float())
        return result["hidden_state"], result["policy_logit"], result["value"].squeeze(-1)

    def recurrent_inference(self, latent, action):
        side = self.task.observation_shape[-1]
        features = torch.nn.functional.one_hot(action.long(), self.task.actions)[:, :-1]
        result = self.network.recurrent_inference(
            latent, features.reshape(-1, 1, side, side).float()
        )
        value = result["value"].squeeze(-1)
        return result["hidden_state"], result["policy_logit"], value, torch.zeros_like(value)
