"""Local snapshots and bounded, non-pickle trajectory messages."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import io
import json
import math
import os
import pickle
import platform
import random
import shutil
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .data import Trajectory

MAX_PAYLOAD = 256 * 1024 * 1024


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Full snapshots written before the package rename contain these records.
        if module == "zero_systems_v3.data":
            module = "zero_systems.data"
        return super().find_class(module, name)


def load_checkpoint(path):
    """Read a trusted local snapshot, including pre-rename record names."""
    record = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        pickle_module=SimpleNamespace(
            __name__="pickle", Unpickler=_CheckpointUnpickler, load=pickle.load
        ),
    )
    if isinstance(record, dict) and record.get("kind") in {"v3-model", "v3-full"}:
        record["kind"] = record["kind"].removeprefix("v3-")
    return record


def source_identity():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def encode_game(game: Trajectory) -> bytes:
    arrays, metadata, chunks = [], {}, []
    for field in dataclasses.fields(game):
        value = getattr(game, field.name)
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biuf":
                raise ValueError("trajectory arrays must be numeric")
            arrays.append((field.name, value.dtype.str, value.shape))
            chunks.append(value.tobytes(order="C"))
        else:
            metadata[field.name] = value
    header = json.dumps(
        {"metadata": metadata, "arrays": arrays}, allow_nan=False, separators=(",", ":")
    ).encode()
    if len(header) > 65536 or 8 + len(header) + sum(map(len, chunks)) > MAX_PAYLOAD:
        raise ValueError("trajectory payload too large")
    return b"".join([struct.pack("<4sI", b"ZSG4", len(header)), header, *chunks])


def decode_game(payload: bytes) -> Trajectory:
    if not 8 <= len(payload) <= MAX_PAYLOAD:
        raise ValueError("invalid trajectory payload size")
    magic, size = struct.unpack_from("<4sI", payload)
    if magic != b"ZSG4" or not 0 < size <= 65536 or 8 + size > len(payload):
        raise ValueError("invalid trajectory header")
    header = json.loads(payload[8 : 8 + size])
    metadata, arrays = header["metadata"], header["arrays"]
    fields = {field.name for field in dataclasses.fields(Trajectory)}
    if not isinstance(metadata, dict) or not isinstance(arrays, list) or len(arrays) > len(fields):
        raise ValueError("invalid trajectory fields")
    offset = 8 + size
    for key, kind, shape in arrays:
        if key not in fields or key in metadata or not isinstance(kind, str):
            raise ValueError("invalid or duplicate trajectory array")
        dtype = np.dtype(kind)
        if (
            dtype.kind not in "biuf"
            or not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        ):
            raise ValueError("invalid trajectory dtype or shape")
        count = math.prod(shape)
        end = offset + count * dtype.itemsize
        if end > len(payload):
            raise ValueError("truncated trajectory array")
        # Views retain the immutable received bytes; replay never mutates them.
        metadata[key] = np.frombuffer(payload, dtype, count, offset).reshape(shape)
        offset = end
    if offset != len(payload):
        raise ValueError("unexpected trailing trajectory bytes")
    return Trajectory(**metadata)


def weight_bytes(model_or_weights) -> bytes:
    weights = (
        model_or_weights.state_dict()
        if isinstance(model_or_weights, torch.nn.Module)
        else model_or_weights
    )
    buffer = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in weights.items()}, buffer)
    return buffer.getvalue()


def decode_weights(payload: bytes):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("weight payload too large")
    result = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(result, dict) or not all(
        isinstance(k, str) and torch.is_tensor(v) for k, v in result.items()
    ):
        raise ValueError("invalid model snapshot")
    return result


def send(connection, message: dict, payload: bytes = b""):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("message too large")
    header = json.dumps({**message, "payload_bytes": len(payload)}, allow_nan=False).encode()
    if len(header) > 65536:
        raise ValueError("message header too large")
    connection.send_bytes(header)
    if payload:
        connection.send_bytes(payload)


def receive(connection):
    message = json.loads(connection.recv_bytes(65536))
    if not isinstance(message, dict) or "payload_bytes" not in message:
        raise ValueError("invalid message header")
    size = message.pop("payload_bytes")
    if not isinstance(size, int) or not 0 <= size <= MAX_PAYLOAD:
        raise ValueError("invalid payload length")
    payload = connection.recv_bytes(MAX_PAYLOAD) if size else b""
    if len(payload) != size:
        raise ValueError("truncated message")
    return message, payload


def atomic_save(path: Path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        torch.save(record, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_state(v) for v in value)
    return value


def rng_state(device):
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None,
    }


def restore_rng(record, device):
    random.setstate(record["python"])
    torch.set_rng_state(record["torch"])
    if record["cuda"] is not None:
        torch.cuda.set_rng_state(record["cuda"], device)


class Journal:
    def __init__(self, directory: Path):
        self.directory = directory
        self.writer = None
        directory.mkdir(parents=True, exist_ok=True)
        self.handle = (directory / "metrics.jsonl").open("a", buffering=1)
        self.started = time.monotonic()

    def recover(self, progress):
        """Retain the old segment and rebuild the active history at a checkpoint."""
        self.handle.close()
        path = self.directory / "metrics.jsonl"
        rows = []
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break  # A killed writer may leave an incomplete final line.
            if (
                row.get("updates", 0) <= progress["updates"]
                and row.get("seconds", 0) <= progress["seconds"]
            ):
                rows.append(row)
            else:
                break
        archive = self.directory / "history" / str(time.time_ns())
        archive.mkdir(parents=True)
        shutil.move(path, archive / path.name)
        path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        for folder in ("models", "checkpoints"):
            for model in (self.directory / folder).glob("step_*.pt"):
                if int(model.stem.removeprefix("step_")) > progress["updates"]:
                    (archive / folder).mkdir(exist_ok=True)
                    shutil.move(model, archive / folder / model.name)
        for result in (self.directory / "evaluations").glob("*/*.json"):
            record = json.loads(result.read_text())
            if (
                record.get("candidate", {}).get("progress", {}).get("updates", 0)
                > progress["updates"]
            ):
                target = archive / result.relative_to(self.directory)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(result, target)
        tensorboard = self.directory / "tensorboard"
        if tensorboard.exists():
            shutil.move(tensorboard, archive / "tensorboard")
        self.handle = path.open("a", buffering=1)

    def enable_tensorboard(self, enabled):
        if not enabled:
            return
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(str(self.directory / "tensorboard" / "training"), flush_secs=30)
        for line in (self.directory / "metrics.jsonl").read_text().splitlines():
            self._mirror(json.loads(line))

    def _mirror(self, record):
        if self.writer is None:
            return
        groups = {
            "progress": (
                "updates",
                "games",
                "positions",
                "learner_samples",
                "seconds",
                "remote_games",
                "remote_positions",
            ),
            "learning": (
                "loss",
                "policy_loss",
                "value_loss",
                "reward_loss",
                "policy_entropy",
                "target_policy_entropy",
                "policy_kl",
                "grad_norm",
                "learning_rate",
            ),
            "replay": (
                "replay_games",
                "replay_positions",
                "sample_model_age_updates",
                "samples_per_position",
                "interval_samples_per_position",
                "target_samples_per_position",
                "available_credit",
                "position_lead_overshoot",
                "games_per_update",
                "target_games_per_update",
            ),
            "self_play": (
                "mean_episode_length",
                "mean_episode_return",
                "termination_fraction",
                "truncation_fraction",
                "resignation_fraction",
                "false_resignation_fraction",
                "resigned_games",
                "resignation_tests",
                "false_resignations",
            ),
            "throughput": (
                "interval_samples_per_second",
                "interval_updates_per_second",
                "interval_positions_per_second",
                "interval_games_per_second",
                "interval_remote_positions_per_second",
                "interval_local_positions_per_second",
            ),
            "timing": (
                "interval_update_seconds",
                "interval_batch_wait_seconds",
                "interval_poll_seconds",
                "interval_decode_seconds",
                "interval_accept_seconds",
                "interval_data_wait_seconds",
                "interval_actor_throttle_seconds",
                "publish_seconds",
                "model_save_seconds",
                "checkpoint_seconds",
            ),
        }
        for group, names in groups.items():
            if record["event"] not in {"learn", "collection", "finished"} and group != "timing":
                continue
            for name in names:
                value = record.get(name)
                if isinstance(value, (float, int)):
                    self.writer.add_scalar(
                        f"{group}/{name}", value, record["updates"], record["time"]
                    )

    def write(self, event: str, **values):
        record = {
            "event": event, "time": time.time(), "monotonic_time": time.monotonic(), **values
        }
        if "policy_entropy" in record:
            record["target_policy_entropy"] = record["policy_entropy"]
        self.handle.write(json.dumps(record, allow_nan=False) + "\n")
        self._mirror(record)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        self.handle.close()


def atomic_json(path: Path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(record, allow_nan=False, indent=2) + "\n")
    os.replace(temporary, path)


def versions():
    result = {
        "source_sha256": source_identity(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
    }
    for name in ("numba", "open_spiel", "gymnasium", "triton"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result
