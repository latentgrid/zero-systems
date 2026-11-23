"""Fixed workloads with warm-up excluded and repeated synchronized timings."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch

from .distributed import LearnerGroup, setup_process
from .env import make_environment
from .model import build_model, initialize
from .play import play_games
from .reference import read_weights
from .replay import Replay, make_batch
from .search import Search
from .storage import versions, weight_bytes


def benchmark(config, system, repeats=5, warmup=3):
    metadata = versions()
    config.validate()
    system.validate()
    if repeats < 2 or warmup < 1:
        raise ValueError("use at least two measurements and one warm-up")
    device = system.learner_devices[0]
    setup_process(system, device, config.seed)
    env = make_environment(config, config.seed)
    position = env.position()
    model = build_model(config, env.task)
    if config.initialize:
        weights, source_task, perspective = read_weights(config.initialize, with_metadata=True)
        if perspective is not None and perspective != config.value_perspective:
            raise ValueError("initialization changes value perspective")
        initialize(model, weights, source_task=source_task)
    group = LearnerGroup(model, config, system, env.task, weight_bytes(model))
    search = Search(
        group.learner.model,
        config,
        device,
        system.actor_precision,
        system.compile,
        system.gpu_search,
    )

    def measure(operation):
        times = []
        for index in range(warmup + repeats):
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            operation()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            if index >= warmup:
                times.append(time.perf_counter() - start)
        return {"seconds": times, "median_seconds": float(np.median(times))}

    try:
        positions = [position] * system.actor_batch
        # Recreate generators so every timed search has identical randomness.
        search_result = measure(
            lambda: search(positions, [np.random.default_rng(i) for i in range(len(positions))])
        )
        replay = Replay(config.replay_games, env.task)
        for game in play_games(search, config, [0, 1], 0):
            replay.add(game)
        sampling = measure(lambda: make_batch(replay, config, np.random.default_rng(0)))
        batch = make_batch(replay, config, np.random.default_rng(0))
        learning = measure(lambda: group.step(batch))
        search_result["roots_per_second"] = system.actor_batch / search_result["median_seconds"]
        learning["samples_per_second"] = config.batch_size / learning["median_seconds"]
        return {
            "experiment": dataclasses.asdict(config),
            "system": dataclasses.asdict(system),
            "versions": metadata,
            "warmup": warmup,
            "repeats": repeats,
            "gpus": [
                torch.cuda.get_device_name(d)
                for d in system.learner_devices
                if d.startswith("cuda")
            ],
            "search": search_result,
            "sampling": sampling,
            "learner": learning,
        }
    finally:
        group.close()
        env.close()
