"""A single optional process watches immutable portable checkpoints."""

from __future__ import annotations

import dataclasses
import fcntl
import json
import multiprocessing as mp
import time
from pathlib import Path

from .config import System, identity
from .evaluate import evaluate, evaluate_episodes, fingerprint
from .storage import atomic_json


def write_tensorboard(directory, name, result):
    from torch.utils.tensorboard import SummaryWriter

    progress = result["candidate"]["progress"]
    with SummaryWriter(str(Path(directory) / "tensorboard" / "evaluations" / name)) as writer:
        for key in (
            "score",
            "mean_return",
            "wins",
            "draws",
            "losses",
            "games",
            "mean_moves",
            "seconds",
            "value_mean",
            "value_std",
            "termination_fraction",
            "truncation_fraction",
            "checkpoint_age_updates",
        ):
            if key in result:
                writer.add_scalar(f"evaluation/{name}/{key}", result[key], progress["updates"])
        bounds = result.get("paired_bootstrap_95", result.get("return_95"))
        for key, value in zip(("lower_95", "upper_95"), bounds, strict=True):
            writer.add_scalar(f"evaluation/{name}/{key}", value, progress["updates"])
        measure = "score" if "score" in result else "mean_return"
        for axis in ("positions", "learner_samples", "seconds"):
            writer.add_scalar(
                f"evaluation/{name}/by_{axis}/{measure}", result[measure], int(progress[axis])
            )


def watch_once(directory, system: System, *, finished=False, seen=None):
    directory = Path(directory)
    seen = set() if seen is None else seen
    models = sorted((directory / "models").glob("step_*.pt"))
    if not models:
        return []
    steps = {p: int(p.stem.removeprefix("step_")) for p in models}
    results = []
    for settings in system.evaluations:
        folder = directory / "evaluations" / settings.name
        protocol = identity(settings)
        previous = []
        for path in folder.glob("*.json"):
            record = json.loads(path.read_text())
            if record.get("evaluation_identity") == protocol and record.get("status") == "complete":
                seen.add(record["job"])
                previous.append(record["candidate"]["progress"]["updates"])
        last = max(previous, default=0)
        eligible = [
            p
            for p in models
            if steps[p] // settings.interval_updates > last // settings.interval_updates
        ]
        if finished and steps[models[-1]] > last:
            eligible.append(models[-1])
        if not eligible:
            continue
        candidate = max(eligible, key=steps.get)
        step = steps[candidate]
        opponent = settings.opponent
        if settings.mode == "self_lag":
            choices = [p for p in models if steps[p] <= step - settings.lag_updates]
            opponent = str(choices[-1]) if choices else None
        candidate_id = fingerprint(candidate)
        try:
            opponent_id = (
                fingerprint(opponent)["sha256"]
                if opponent not in (None, "random") and settings.mode != "episodes"
                else str(opponent)
            )
        except OSError:
            # Evaluation below records this job's missing opponent without
            # preventing independent configured evaluations from running.
            opponent_id = str(opponent)
        job = f"{protocol}:{candidate_id['sha256']}:{opponent_id}"
        if job in seen:
            continue
        seen.add(job)
        status_path = directory / "evaluations" / "status.json"
        atomic_json(
            status_path,
            {"status": "running", "name": settings.name, "updates": step, "time": time.time()},
        )
        try:
            if settings.mode == "self_lag" and opponent is None:
                result = {
                    "status": "skipped",
                    "reason": "no checkpoint at requested lag",
                    "candidate": {**candidate_id, "progress": {"updates": step}},
                }
            else:
                kwargs = dict(
                    device=settings.device,
                    batch=settings.batch,
                    simulations=settings.simulations,
                    seed=settings.seed,
                    search=settings.search,
                    precision=settings.precision,
                )
                if settings.mode == "episodes":
                    result = evaluate_episodes(candidate, episodes=settings.games, **kwargs)
                else:
                    result = evaluate(
                        candidate,
                        opponent,
                        games=settings.games,
                        opening_moves=settings.opening_moves,
                        **kwargs,
                    )
                result["status"] = "complete"
                latest = max(
                    (
                        int(p.stem.removeprefix("step_"))
                        for p in (directory / "models").glob("step_*.pt")
                    ),
                    default=step,
                )
                result["checkpoint_age_updates"] = latest - step
        except Exception as error:
            result = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "candidate": {**candidate_id, "progress": {"updates": step}},
            }
        result.update(
            job=job,
            evaluation_identity=protocol,
            evaluation=dataclasses.asdict(settings),
            skipped_checkpoints=sorted({steps[p] for p in eligible if p != candidate}),
        )
        path = folder / f"step_{step:08d}_{protocol[:12]}.json"
        atomic_json(path, result)
        atomic_json(
            status_path,
            {
                "status": result["status"],
                "name": settings.name,
                "updates": step,
                "error": result.get("error"),
                "time": time.time(),
            },
        )
        if system.tensorboard and result["status"] == "complete":
            write_tensorboard(directory, settings.name, result)
        results.append(result)
    return results


def training_finished(directory):
    path = Path(directory) / "metrics.jsonl"
    if not path.exists():
        return False
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 8192))
        lines = handle.read().splitlines()
    try:
        return bool(lines) and json.loads(lines[-1]).get("event") in {"finished", "failed"}
    except json.JSONDecodeError:
        return False


def evaluation_status(directory, system):
    results = {}
    for settings in system.evaluations:
        records = [
            json.loads(p.read_text())
            for p in (Path(directory) / "evaluations" / settings.name).glob("*.json")
        ]
        records = [r for r in records if r.get("evaluation_identity") == identity(settings)]
        latest = max(records, key=lambda r: r["candidate"]["progress"]["updates"], default=None)
        results[settings.name] = (
            {
                "status": latest["status"],
                "updates": latest["candidate"]["progress"]["updates"],
                "error": latest.get("error"),
            }
            if latest
            else {"status": "pending"}
        )
    path = Path(directory) / "evaluations/status.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state["status"] == "failed":
            results["watcher"] = state
    return results


def watch(directory, system: System, done=None, *, once=False):
    directory = Path(directory)
    folder = directory / "evaluations"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "watch.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("this run already has a live evaluator") from error
        for settings in system.evaluations:
            if (
                system.tensorboard
                and not (directory / "tensorboard" / "evaluations" / settings.name).exists()
            ):
                for path in sorted((folder / settings.name).glob("*.json")):
                    result = json.loads(path.read_text())
                    if result.get("status") == "complete":
                        write_tensorboard(directory, settings.name, result)
        seen = set()
        while True:
            finished = (done is not None and done.is_set()) or training_finished(directory)
            watch_once(directory, system, finished=finished or once, seen=seen)
            if finished or once:
                return
            if done is not None:
                done.wait(5)
            else:
                time.sleep(5)


def _watch_process(directory, system, done):
    try:
        watch(directory, system, done)
    except Exception as error:
        atomic_json(
            Path(directory) / "evaluations" / "status.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "time": time.time(),
            },
        )


def start_watcher(directory, system):
    context = mp.get_context("spawn")
    done = context.Event()
    process = context.Process(target=_watch_process, args=(str(directory), system, done))
    process.start()
    return process, done
