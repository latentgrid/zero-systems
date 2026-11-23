"""Immutable checkpoint evaluation shared by the CLI and live watcher."""

from __future__ import annotations

import dataclasses
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from .config import Experiment, from_dict
from .env import environment_settings, make_environment, task_matches, task_spec
from .model import build_model
from .reference import MiniZeroReference
from .search import Search
from .storage import load_checkpoint, versions


def load_candidate(path, device):
    record = load_checkpoint(path)
    if record.get("kind") not in {"model", "full"}:
        raise ValueError("candidate must be a Zero Systems checkpoint")
    config = from_dict(record["experiment"], Experiment)
    task = task_spec(config)
    if not task_matches(record["task"], task, config):
        raise ValueError(
            "checkpoint observation/action encoding is incompatible with this environment"
        )
    model = build_model(config, task).to(device)
    model.load_state_dict(record["model"])
    return model, config, record["progress"]


def fingerprint(path):
    with Path(path).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(Path(path).resolve()), "sha256": digest}


def interval(values, seed):
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, (4000, len(values))), axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def _episodes(config, searches, count, batch, seed, opening_moves):
    competitive = len(searches) == 2
    width = batch - batch % 2 if competitive else batch
    returns, lengths, terminated, predictions = [], [], [], []
    for start in range(0, count, width):
        ids = list(range(start, min(count, start + width)))
        envs = []
        try:
            for i in ids:
                envs.append(make_environment(config, seed + (i // 2 if competitive else i)))
            rngs = [
                np.random.default_rng(np.random.SeedSequence([seed, i // 2 if competitive else i]))
                for i in ids
            ]
            totals = np.zeros((len(ids), envs[0].task.players), np.float64)
            moves = np.zeros(len(ids), np.int32)
            for row, env in enumerate(envs):
                for _ in range(opening_moves):
                    if env.terminated or env.truncated:
                        break
                    legal = env.position().legal.copy()
                    if config.environment == "go":
                        legal[-1] = False
                    choices = np.flatnonzero(legal)
                    if not len(choices):
                        break
                    totals[row] += env.step(int(rngs[row].choice(choices)))
                    moves[row] += 1
            active = [i for i, e in enumerate(envs) if not e.terminated and not e.truncated]
            while active:
                positions = {i: envs[i].position() for i in active}
                for network, search in enumerate(searches):
                    selected = [
                        i
                        for i in active
                        if not competitive or int(positions[i].player != ids[i] % 2) == network
                    ]
                    if not selected:
                        continue
                    if search is None:
                        actions = [
                            int(rngs[i].choice(np.flatnonzero(positions[i].legal)))
                            for i in selected
                        ]
                    else:
                        result = search(
                            [positions[i] for i in selected],
                            [rngs[i] for i in selected],
                            explore=False,
                        )
                        actions = result.actions
                        if network == 0:
                            predictions.extend(result.value.tolist())
                    for row, action in zip(selected, actions, strict=True):
                        totals[row] += envs[row].step(int(action))
                        moves[row] += 1
                active = [i for i in active if not envs[i].terminated and not envs[i].truncated]
            for row, identity in enumerate(ids):
                if envs[row].outcome is not None:
                    totals[row] += envs[row].outcome
                seat = identity % 2 if competitive else 0
                returns.append(float(totals[row, seat]))
                lengths.append(int(moves[row]))
                terminated.append(bool(envs[row].terminated))
        finally:
            for env in envs:
                env.close()
    return {
        "returns": returns,
        "mean_return": float(np.mean(returns)),
        "mean_moves": float(np.mean(lengths)),
        "episode_lengths": lengths,
        "termination_fraction": float(np.mean(terminated)),
        "truncation_fraction": 1 - float(np.mean(terminated)),
        "value_mean": float(np.mean(predictions)) if predictions else 0.0,
        "value_std": float(np.std(predictions)) if predictions else 0.0,
    }


def evaluate(
    candidate,
    opponent,
    device="cpu",
    games=200,
    batch=64,
    simulations=16,
    seed=303,
    opening_moves=2,
    *,
    search=None,
    precision="float32",
):
    if games <= 0 or games % 2 or batch < 2 or simulations <= 0 or opening_moves < 0:
        raise ValueError(
            "evaluation requires positive even games, batch >= 2 and valid search/openings"
        )
    return _evaluate(
        candidate,
        opponent,
        device,
        games,
        batch,
        simulations,
        seed,
        opening_moves,
        search,
        precision,
    )


def evaluate_episodes(
    candidate,
    device="cpu",
    episodes=30,
    batch=32,
    simulations=16,
    seed=7301,
    *,
    search=None,
    precision="float32",
):
    if min(episodes, batch, simulations) <= 0:
        raise ValueError("episode evaluation counts must be positive")
    return _evaluate(
        candidate, None, device, episodes, batch, simulations, seed, 0, search, precision
    )


def _evaluate(
    candidate, opponent, device, count, batch, simulations, seed, openings, algorithm, precision
):
    if (
        algorithm not in {None, "puct", "gumbel"}
        or precision not in {"float32", "bfloat16"}
        or seed < 0
    ):
        raise ValueError("invalid evaluation search, precision or seed")
    torch.set_num_threads(1)
    started = time.monotonic()
    candidate_id = fingerprint(candidate)
    model, config, progress = load_candidate(candidate, device)
    competitive = opponent is not None
    if model.task.players != (2 if competitive else 1):
        raise ValueError(
            "use paired evaluation for two players and episode evaluation for one player"
        )
    config = dataclasses.replace(config, simulations=simulations, search=algorithm or config.search)
    searches = [Search(model, config, device, precision)]
    opponent_id = None
    if competitive:
        if str(opponent) == "random":
            searches.append(None)
            opponent_id = {"path": "random", "progress": None}
        else:
            opponent_id = fingerprint(opponent)
            try:
                other = MiniZeroReference(opponent, device)
                if config.environment != "go":
                    raise ValueError("MiniZero reference requires Go")
                other_config, other_progress = (
                    dataclasses.replace(config, value_perspective="player0"),
                    None,
                )
            except (RuntimeError, ValueError):
                other, other_config, other_progress = load_candidate(opponent, device)
                if environment_settings(config) != environment_settings(other_config):
                    raise ValueError("evaluation environments or rules differ") from None
                other_config = dataclasses.replace(
                    config, value_perspective=other_config.value_perspective
                )
            if model.task != other.task:
                raise ValueError("evaluation models use incompatible observation/action encodings")
            searches.append(Search(other, other_config, device, precision))
            opponent_id["progress"] = other_progress
    result = _episodes(config, searches, count, batch, seed, openings)
    if fingerprint(candidate) != candidate_id or (
        opponent_id
        and str(opponent) != "random"
        and fingerprint(opponent)["sha256"] != opponent_id["sha256"]
    ):
        raise RuntimeError("checkpoint changed during evaluation")
    result.update(
        candidate={**candidate_id, "progress": progress},
        opponent=opponent_id,
        protocol="Zero Systems common-search diagnostic; not the native MiniZero executable"
        if competitive
        else "Zero Systems fixed-seed episode returns",
        experiment=dataclasses.asdict(config),
        versions=versions(),
        device=str(device),
        precision=precision,
        seed=seed,
        games=count,
        opening_moves=openings,
        seconds=time.monotonic() - started,
    )
    if competitive:
        utilities = np.asarray(result["returns"])
        scores = np.where(np.isclose(utilities, 0, atol=1e-8), 0.5, (utilities > 0).astype(float))
        pairs = scores.reshape(-1, 2).mean(1)
        result.update(
            wins=int(np.sum(scores == 1)),
            draws=int(np.sum(scores == 0.5)),
            losses=int(np.sum(scores == 0)),
            score=float(scores.mean()),
            pair_scores=pairs.tolist(),
            paired_bootstrap_95=interval(pairs, seed),
        )
    else:
        result["return_95"] = interval(result["returns"], seed)
    return result
