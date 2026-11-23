"""Persistent DDP replicas; the coordinator alone samples replay."""

from __future__ import annotations

import datetime
import multiprocessing as mp
import os
import random
import socket
import time
import traceback

import torch

from .data import Batch
from .learn import Learner
from .model import build_model
from .storage import decode_weights, restore_rng, rng_state


def setup_process(system, device, seed):
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    if system.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.set_num_threads(system.threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(system.deterministic)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = not system.deterministic


def _replica(rank, config, system, task, payload, optimizer, endpoint, connection, saved_rng):
    try:
        device = system.learner_devices[rank]
        setup_process(system, device, config.seed + rank)
        torch.distributed.init_process_group(
            "nccl",
            init_method=endpoint,
            rank=rank,
            world_size=len(system.learner_devices),
            timeout=datetime.timedelta(seconds=system.timeout),
        )
        model = build_model(config, task)
        model.load_state_dict(decode_weights(payload))
        learner = Learner(model, config, system, device, distributed=True)
        if optimizer:
            learner.optimizer.load_state_dict(optimizer)
        if saved_rng:
            restore_rng(saved_rng, device)
        connection.send(("ready", None))
        shared = None
        while True:
            command, value = connection.recv()
            if command == "close":
                break
            if command == "batch":
                shared = value
            elif command == "step":
                learner.step(shared)
                connection.send(("done", None))
            elif command == "rng":
                connection.send(("rng", rng_state(device)))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class LearnerGroup:
    def __init__(self, model, config, system, task, payload, optimizer=None, saved_rng=None):
        self.system = system
        self.processes, self.connections = [], []
        self.shared = None
        self.distributed = len(system.learner_devices) > 1
        try:
            device = system.learner_devices[0]
            setup_process(system, device, config.seed)
            if self.distributed:
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                endpoint = f"tcp://127.0.0.1:{port}"
                context = mp.get_context("spawn")
                torch.multiprocessing.set_sharing_strategy("file_system")
                for rank in range(1, len(system.learner_devices)):
                    parent, child = context.Pipe()
                    process = context.Process(
                        target=_replica,
                        args=(
                            rank,
                            config,
                            system,
                            task,
                            payload,
                            optimizer,
                            endpoint,
                            child,
                            saved_rng[rank] if saved_rng else None,
                        ),
                    )
                    process.start()
                    child.close()
                    self.processes.append(process)
                    self.connections.append(parent)
                torch.distributed.init_process_group(
                    "nccl",
                    init_method=endpoint,
                    rank=0,
                    world_size=len(system.learner_devices),
                    timeout=datetime.timedelta(seconds=system.timeout),
                )
            self.learner = Learner(model, config, system, device, distributed=self.distributed)
            if optimizer:
                self.learner.optimizer.load_state_dict(optimizer)
            if saved_rng:
                restore_rng(saved_rng[0], device)
            for connection in self.connections:
                self._receive(connection, "ready")
        except BaseException:
            self.close()
            raise

    def _receive(self, connection, expected):
        deadline = time.monotonic() + self.system.timeout
        while not connection.poll(0.1):
            if time.monotonic() > deadline or any(p.exitcode is not None for p in self.processes):
                raise RuntimeError("learner replica stopped responding")
        kind, value = connection.recv()
        if kind != expected:
            raise RuntimeError(f"learner replica {kind}: {value}")
        return value

    def step(self, batch):
        if self.distributed:
            if self.shared is None:
                self.shared = Batch(
                    **{k: v.clone().share_memory_() for k, v in vars(batch).items()}
                )
                for connection in self.connections:
                    connection.send(("batch", self.shared))
            else:
                for key, value in vars(batch).items():
                    getattr(self.shared, key).copy_(value)
            for connection in self.connections:
                connection.send(("step", None))
            result = self.learner.step(self.shared)
            for connection in self.connections:
                self._receive(connection, "done")
            return result
        return self.learner.step(batch)

    def rng(self):
        for connection in self.connections:
            connection.send(("rng", None))
        return [rng_state(self.system.learner_devices[0])] + [
            self._receive(c, "rng") for c in self.connections
        ]

    def close(self):
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (OSError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for connection in self.connections:
            connection.close()
        if self.distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
