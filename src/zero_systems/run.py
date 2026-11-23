"""The training workflow: collect, learn, publish, checkpoint.

The asynchronous loop uses the same owners and operations with bounded credits.
Only this coordinator accepts game identities and changes training progress.
"""

from __future__ import annotations

import dataclasses
import signal
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .config import Experiment, System, from_dict, learning_settings
from .distributed import LearnerGroup
from .env import task_matches, task_spec
from .model import build_model, initialize
from .play import play_games
from .reference import read_weights
from .replay import Replay, make_batch
from .search import Search
from .storage import (
    Journal,
    atomic_json,
    atomic_save,
    cpu_state,
    decode_game,
    load_checkpoint,
    source_identity,
    versions,
    weight_bytes,
)
from .workers import ActorPool


class AcceptedGames:
    """A retired prefix plus out-of-order completions; independent of replay eviction."""

    def __init__(self, prefix=0):
        self.prefix = prefix
        self.pending = set()

    def accept(self, identity):
        if identity < self.prefix or identity in self.pending:
            return False
        self.pending.add(identity)
        while self.prefix in self.pending:
            self.pending.remove(self.prefix)
            self.prefix += 1
        return True


class TrainingRun:
    def __init__(
        self,
        experiment: Experiment,
        system: System,
        directory,
        resume=None,
        components=("representation", "dynamics", "prediction"),
    ):
        experiment.validate()
        system.validate()
        self.experiment, self.system = experiment, system
        self.directory = Path(directory)
        existing = (self.directory / "metrics.jsonl").exists() or (
            self.directory / "models"
        ).exists()
        if not resume and existing:
            raise ValueError("run directory already exists; resume it or choose a new directory")
        if (
            resume
            and existing
            and Path(resume).resolve().parent != (self.directory / "checkpoints").resolve()
        ):
            raise ValueError("use a new run directory when resuming another run's checkpoint")
        self.source_sha256 = source_identity()
        self.started = time.monotonic()
        self.previous_seconds = 0.0
        self.stop_requested = False
        self.generation = uuid.uuid4().hex
        self.task = task_spec(experiment)
        self.replay = Replay(experiment.replay_games, self.task)
        self.rng = np.random.default_rng(np.random.SeedSequence([experiment.seed, 98731]))
        self.accepted = AcceptedGames()
        self.next_game = 0
        self.games = self.positions = 0
        self.remote_games = self.remote_positions = 0
        self.resigned_games = self.resignation_tests = self.false_resignations = 0
        self.terminated_games = self.truncated_games = 0
        self.return_sum = 0.0
        self.phase = "collect"
        self.round_end = experiment.games_per_round
        self.remaining_updates = 0
        self.jobs = {}
        self.staged = {}
        self.pending_batch = None
        self.pool = self.group = None
        self.last_full = self.last_model = 0
        self.journal = Journal(self.directory)
        self.last_log_time = time.monotonic()
        self.last_log_samples = 0
        self.interval_update_seconds = self.interval_batch_wait_seconds = 0.0
        self.interval_data_wait_seconds = self.interval_actor_throttle_seconds = 0.0
        self.last_log_positions = self.last_log_games = self.last_log_remote_positions = 0
        self.interval_poll_seconds = self.interval_decode_seconds = self.interval_accept_seconds = (
            0.0
        )
        torch.manual_seed(experiment.seed)
        model = build_model(experiment, self.task)
        optimizer, rngs = None, None
        restored = None
        if resume:
            restored = load_checkpoint(resume)
            if restored.get("kind") != "full":
                raise ValueError("resume requires a complete training checkpoint")
            if not task_matches(restored["task"], self.task, experiment):
                raise ValueError("resume changes observation/action encoding")
            old = learning_settings(from_dict(restored["experiment"], Experiment))
            new = learning_settings(experiment)
            if old != new:
                raise ValueError(
                    "resume changes learning settings; use initialization for a new run"
                )
            model.load_state_dict(restored["model"])
            optimizer, rngs = restored["optimizer"], restored["rank_rng"]
            if len(rngs) != len(system.learner_devices):
                raise ValueError("resume requires the same learner rank count")
            for game in restored["replay"]:
                self.replay.add(game)
            self.rng.bit_generator.state = restored["sampling_rng"]
            self.pending_batch = restored.get("pending_batch")
            for key in (
                "next_game",
                "games",
                "positions",
                "phase",
                "round_end",
                "remaining_updates",
            ):
                setattr(self, key, restored["progress"][key])
            self.accepted = AcceptedGames(self.next_game)
            self.remote_games = restored["progress"].get("remote_games", 0)
            self.remote_positions = restored["progress"].get("remote_positions", 0)
            for key in ("resigned_games", "resignation_tests", "false_resignations"):
                setattr(self, key, restored["progress"].get(key, 0))
            self.previous_seconds = restored["progress"]["seconds"]
            self.published = restored["published"]
            self.published_step = restored["published_step"]
            for key in ("terminated_games", "truncated_games", "return_sum"):
                setattr(self, key, restored["progress"].get(key, 0))
            self.journal.recover(restored["progress"])
        elif experiment.initialize:
            weights, source_task, perspective = read_weights(
                experiment.initialize, with_metadata=True
            )
            if perspective is not None and perspective != experiment.value_perspective:
                raise ValueError("initialization changes value perspective")
            report = initialize(model, weights, components, source_task=source_task)
            self.journal.write("initialize", source=experiment.initialize, **report)
        payload = weight_bytes(model)
        try:
            self.journal.enable_tensorboard(system.tensorboard)
            self.group = LearnerGroup(
                model, experiment, system, self.task, payload, optimizer, rngs
            )
            self.learner = self.group.learner
            if restored:
                self.learner.updates = restored["progress"]["updates"]
                self.learner.samples = restored["progress"]["learner_samples"]
                self.last_log_samples = self.learner.samples
                self.last_full = self.last_model = self.learner.updates
                self.last_log_positions, self.last_log_games = self.positions, self.games
                self.last_log_remote_positions = self.remote_positions
            else:
                self.published, self.published_step = payload, 0
            self.inline = (
                experiment.schedule == "phased" and not system.actor_devices and not system.listen
            )
            if not self.inline:
                actors = system.actor_devices or (
                    () if system.listen else (system.learner_devices[0],)
                )
                self.pool = ActorPool(experiment, dataclasses.replace(system, actor_devices=actors))
                self.pool.wait_local()
            else:
                self.search = Search(
                    self.learner.model,
                    experiment,
                    system.learner_devices[0],
                    system.actor_precision,
                    system.compile,
                    system.gpu_search,
                )
            self.journal.write(
                "start",
                experiment=dataclasses.asdict(experiment),
                system=dataclasses.asdict(system),
                versions=versions(),
                source_sha256=self.source_sha256,
                resume=str(resume) if resume else None,
                resumed_seconds=self.previous_seconds,
            )
            self.save_model(force=True)
        except BaseException:
            if self.pool:
                self.pool.close()
            if self.group:
                self.group.close()
            self.journal.close()
            raise

    def progress(self):
        return {
            "updates": self.learner.updates,
            "learner_samples": self.learner.samples,
            "games": self.games,
            "positions": self.positions,
            "samples_per_position": self.learner.samples / self.positions
            if self.positions
            else None,
            "position_lead_overshoot": max(
                0.0,
                float(
                    self.positions
                    - self.learner.samples / self.experiment.position_ratio
                    - self.experiment.position_lead
                ),
            )
            if self.experiment.position_ratio
            else 0.0,
            "remote_games": self.remote_games,
            "remote_positions": self.remote_positions,
            "resigned_games": self.resigned_games,
            "resignation_tests": self.resignation_tests,
            "false_resignations": self.false_resignations,
            "terminated_games": self.terminated_games,
            "truncated_games": self.truncated_games,
            "return_sum": self.return_sum,
            "next_game": self.next_game,
            "phase": self.phase,
            "round_end": self.round_end,
            "remaining_updates": self.remaining_updates,
            "seconds": self.previous_seconds + time.monotonic() - self.started,
        }

    def stopping(self):
        if self.system.max_seconds and self.progress()["seconds"] >= self.system.max_seconds:
            self.stop_requested = True
        return self.stop_requested or self.learner.updates >= self.experiment.updates

    def accept(self, game, *, remote=False):
        game.validate(self.task)
        if not self.accepted.accept(game.game_id):
            return False
        self.games += 1
        self.positions += len(game)
        self.terminated_games += int(game.terminated)
        self.truncated_games += int(game.truncated)
        self.return_sum += float(game.rewards[:, 0].sum()) + (
            float(game.outcome[0]) if game.outcome is not None else 0.0
        )
        self.resigned_games += int(game.resigned_player >= 0)
        if game.resignation_test_player >= 0:
            self.resignation_tests += 1
            self.false_resignations += int(game.outcome[game.resignation_test_player] >= 0)
        if remote:
            self.remote_games += 1
            self.remote_positions += len(game)
        if self.experiment.schedule == "phased":
            self.staged[game.game_id] = game
        else:
            self.replay.add(game)
        return True

    def _dispatch(self, limit):
        available = self.pool.available()
        slots = len(available)
        for peer in available:
            count = min(peer.capacity, max(0, (limit - self.next_game + slots - 1) // slots))
            slots -= 1
            if count == 0:
                continue
            ids = list(range(self.next_game, self.next_game + count))
            self.next_game += count
            token = f"{self.generation}:{ids[0]}:0"
            work = {"kind": "work", "token": token, "game_ids": ids, "step": self.published_step}
            self.jobs[token] = {
                "work": work,
                "received": set(),
                "payload": self.published,
                "attempts": 0,
                "peer": peer,
            }
            peer.submit(work, self.published)

    def _retry_pending(self):
        for job in list(self.jobs.values()):
            if job["peer"] is not None:
                continue
            available = self.pool.available()
            if not available:
                return
            ids = job["work"]["game_ids"]
            peer = available[0]
            if len(ids) > peer.capacity:
                remaining = ids[peer.capacity :]
                token = f"{self.generation}:{remaining[0]}:{job['attempts']}"
                self.jobs[token] = {
                    **job,
                    "received": set(),
                    "work": {**job["work"], "token": token, "game_ids": remaining},
                }
                job["work"] = {**job["work"], "game_ids": ids[: peer.capacity]}
            job["peer"] = peer
            peer.submit(job["work"], job["payload"])

    def _receive(self, timeout=0.1):
        start = time.monotonic()
        result = self.pool.poll(timeout)
        self.interval_poll_seconds += time.monotonic() - start
        if result is None:
            return
        peer, message, payload = result
        kind = message["kind"]
        if kind == "ready":
            return
        work = peer.active
        if work is None:
            if kind == "error":
                self.pool.retire(peer)
                self.journal.write("actor_failure", error=message["error"], missing=0)
            return
        token = message.get("token", work["token"])
        job = self.jobs.get(token)
        if job is None or job["peer"] is not peer:
            return
        if kind == "game":
            start = time.monotonic()
            game = decode_game(payload)
            self.interval_decode_seconds += time.monotonic() - start
            if game.game_id != message["game_id"]:
                raise ValueError("trajectory identity disagrees with its header")
            if game.game_id not in work["game_ids"] or game.generating_update != work["step"]:
                raise ValueError("actor returned an unassigned game or incorrect model version")
            start = time.monotonic()
            self.accept(game, remote=peer.device is None)
            self.interval_accept_seconds += time.monotonic() - start
            job["received"].add(game.game_id)
        elif kind == "done":
            if job["received"] != set(work["game_ids"]):
                raise RuntimeError("actor completed a job without delivering all games")
            del self.jobs[token]
            peer.active = None
            peer.ready = True
        elif kind == "error":
            self.pool.retire(peer)
            del self.jobs[token]
            missing = sorted(set(work["game_ids"]) - job["received"])
            self.journal.write("actor_failure", error=message["error"], missing=len(missing))
            if missing:
                attempts = job["attempts"] + 1
                if attempts > 2:
                    raise RuntimeError("actor assignment failed after two retries")
                new_token = f"{self.generation}:{missing[0]}:{attempts}"
                job.update(
                    work={**work, "token": new_token, "game_ids": missing},
                    received=set(),
                    attempts=attempts,
                    peer=None,
                )
                self.jobs[new_token] = job

    def _drain(self):
        deadline = time.monotonic() + self.system.timeout
        while self.jobs:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "could not drain actor work; previous full checkpoint remains usable"
                )
            self._retry_pending()
            self._receive()

    def collect(self):
        if self.inline:
            while self.games < self.round_end and not self.stopping():
                count = min(self.system.actor_batch, self.round_end - self.next_game)
                ids = list(range(self.next_game, self.next_game + count))
                self.next_game += count
                play_games(self.search, self.experiment, ids, self.published_step, emit=self.accept)
        else:
            last_progress = time.monotonic()
            previous = self.games
            while self.games < self.round_end and not self.stopping():
                self._retry_pending()
                self._dispatch(self.round_end)
                self._receive()
                if self.games != previous:
                    previous, last_progress = self.games, time.monotonic()
                elif time.monotonic() - last_progress > self.system.timeout:
                    raise TimeoutError("no self-play progress")
            self._drain()
        for identity in sorted(self.staged):
            self.replay.add(self.staged[identity])
        self.staged.clear()
        if self.games >= self.round_end:
            self.phase = "learn"
            self.remaining_updates = self.experiment.updates_per_round
        self.journal.write("collection", **self.progress(), replay_positions=self.replay.positions)

    def update(self):
        batch = self.pending_batch
        if batch is None:
            batch = make_batch(self.replay, self.experiment, self.rng)
        future = (
            self.batch_pool.submit(make_batch, self.replay, self.experiment, self.rng)
            if self.experiment.schedule == "async" or self.remaining_updates > 1
            else None
        )
        start = time.monotonic()
        metrics = self.group.step(batch)
        update_seconds = time.monotonic() - start
        self.interval_update_seconds += update_seconds
        # Finish preparation before collection can mutate replay. Keep the prepared
        # batch with its advanced sampling RNG so a checkpoint resumes exactly.
        wait_start = time.monotonic()
        self.pending_batch = future.result() if future else None
        self.interval_batch_wait_seconds += time.monotonic() - wait_start
        if self.learner.updates % self.system.log_updates == 0:
            entropy = (-torch.xlogy(batch.policy, batch.policy).sum(-1) * batch.valid).mean(0)
            policy_entropy = float(entropy[0] + entropy[1:].mean())
            now = time.monotonic()
            interval = now - self.last_log_time
            rate = (self.learner.samples - self.last_log_samples) / interval
            self.journal.write(
                "learn",
                **self.progress(),
                loss=metrics["loss"],
                policy_loss=metrics["policy_loss"],
                value_loss=metrics["value_loss"],
                reward_loss=metrics["reward_loss"],
                policy_entropy=policy_entropy,
                policy_kl=metrics["policy_loss"] - policy_entropy,
                grad_norm=metrics["grad_norm"],
                learning_rate=self.experiment.learning_rate,
                games_per_update=self.games / self.learner.updates,
                target_games_per_update=float(self.experiment.game_ratio)
                if not self.experiment.position_ratio else None,
                update_seconds=update_seconds,
                interval_samples_per_second=rate,
                interval_updates_per_second=rate / self.experiment.batch_size,
                interval_positions_per_second=(self.positions - self.last_log_positions) / interval,
                interval_games_per_second=(self.games - self.last_log_games) / interval,
                interval_remote_positions_per_second=(
                    self.remote_positions - self.last_log_remote_positions
                )
                / interval,
                interval_local_positions_per_second=(
                    self.positions
                    - self.last_log_positions
                    - self.remote_positions
                    + self.last_log_remote_positions
                )
                / interval,
                interval_samples_per_position=(self.learner.samples - self.last_log_samples)
                / (self.positions - self.last_log_positions)
                if self.positions > self.last_log_positions
                else None,
                target_samples_per_position=self.experiment.samples_per_position,
                available_credit=float(
                    self.experiment.position_ratio * self.positions - self.learner.samples
                )
                if self.experiment.position_ratio
                else float(
                    self.games / self.experiment.game_ratio * self.experiment.batch_size
                    - self.learner.samples
                ),
                mean_episode_length=self.positions / max(1, self.games),
                mean_episode_return=self.return_sum / max(1, self.games),
                termination_fraction=self.terminated_games / max(1, self.games),
                truncation_fraction=self.truncated_games / max(1, self.games),
                resignation_fraction=self.resigned_games / max(1, self.games),
                false_resignation_fraction=self.false_resignations / max(1, self.resignation_tests),
                interval_update_seconds=self.interval_update_seconds,
                interval_batch_wait_seconds=self.interval_batch_wait_seconds,
                interval_poll_seconds=self.interval_poll_seconds,
                interval_decode_seconds=self.interval_decode_seconds,
                interval_accept_seconds=self.interval_accept_seconds,
                interval_data_wait_seconds=self.interval_data_wait_seconds,
                interval_actor_throttle_seconds=self.interval_actor_throttle_seconds,
                sample_model_age_updates=float(
                    self.learner.updates - batch.model_steps.float().mean()
                ),
                replay_positions=self.replay.positions,
                replay_games=len(self.replay.games),
            )
            self.last_log_time, self.last_log_samples = now, self.learner.samples
            self.last_log_positions, self.last_log_games = self.positions, self.games
            self.last_log_remote_positions = self.remote_positions
            self.interval_data_wait_seconds = self.interval_actor_throttle_seconds = 0.0
            self.interval_update_seconds = self.interval_batch_wait_seconds = 0.0
            self.interval_poll_seconds = self.interval_decode_seconds = 0.0
            self.interval_accept_seconds = 0.0

    def publish(self):
        started = time.monotonic()
        self.published = weight_bytes(self.learner.model)
        self.published_step = self.learner.updates
        self.journal.write("publish", **self.progress(), publish_seconds=time.monotonic() - started)

    def save_model(self, force=False):
        if not force and self.stopping():
            return
        if not force and self.learner.updates - self.last_model < self.system.model_updates:
            return
        path = self.directory / "models" / f"step_{self.learner.updates:08d}.pt"
        if path.exists():
            self.last_model = self.learner.updates
            return
        started = time.monotonic()
        record = {
            "kind": "model",
            "format": 1,
            "model": cpu_state(self.learner.model.state_dict()),
            "experiment": dataclasses.asdict(self.experiment),
            "task": dataclasses.asdict(self.task),
            "progress": self.progress(),
            "system": dataclasses.asdict(self.system),
            "source_sha256": self.source_sha256,
        }
        atomic_save(path, record)
        self.last_model = self.learner.updates
        self.journal.write(
            "model_available", **self.progress(), model_path=str(path),
            model_save_seconds=time.monotonic() - started,
        )

    def checkpoint(self):
        if (
            self.jobs
            or self.staged
            or self.accepted.pending
            or self.accepted.prefix != self.next_game
        ):
            raise RuntimeError("full checkpoint requires a quiescent, fully accepted work boundary")
        started = time.monotonic()
        record = {
            "kind": "full",
            "format": 1,
            "model": cpu_state(self.learner.model.state_dict()),
            "optimizer": cpu_state(self.learner.optimizer.state_dict()),
            "rank_rng": self.group.rng(),
            "sampling_rng": self.rng.bit_generator.state,
            "pending_batch": self.pending_batch,
            "published": self.published,
            "published_step": self.published_step,
            "experiment": dataclasses.asdict(self.experiment),
            "system": dataclasses.asdict(self.system),
            "task": dataclasses.asdict(self.task),
            "progress": self.progress(),
            "replay": list(self.replay.games.values()),
            "source_sha256": self.source_sha256,
        }
        folder = self.directory / "checkpoints"
        atomic_save(folder / f"step_{self.learner.updates:08d}.pt", record)
        for old in sorted(folder.glob("step_*.pt"))[:-2]:
            old.unlink()
        self.last_full = self.learner.updates
        self.journal.write(
            "checkpoint", **self.progress(), checkpoint_seconds=time.monotonic() - started
        )

    def phased(self):
        while not self.stopping():
            if self.phase == "collect":
                self.collect()
            while self.phase == "learn" and self.remaining_updates and not self.stopping():
                self.update()
                self.remaining_updates -= 1
                self.save_model()
            if self.phase == "learn" and not self.remaining_updates:
                self.publish()
                self.phase = "collect"
                self.round_end = self.games + self.experiment.games_per_round
            if self.learner.updates - self.last_full >= self.system.checkpoint_updates:
                self.checkpoint()

    def asynchronous(self):
        config = self.experiment
        ratio = config.game_ratio
        last_progress = time.monotonic()
        while not self.stopping():
            before = (self.games, self.learner.updates)
            started = time.monotonic()
            self._retry_pending()
            if config.position_ratio is not None:
                headroom = (
                    self.learner.samples / config.position_ratio
                    + config.position_lead
                    - self.positions
                )
                # Complete games can overshoot the soft position bound. Reserve one
                # decision per in-flight game and cap all work through the actor pool.
                limit = self.next_game + max(0, int(headroom) - (self.next_game - self.games))
            else:
                limit = min(
                    int(self.learner.updates * ratio) + config.game_lead,
                    int(config.updates * ratio),
                )
            throttled = limit <= self.next_game
            self._dispatch(limit)
            self._receive(timeout=0.001)
            if config.can_update(
                self.games, self.positions, self.learner.samples, self.learner.updates
            ):
                self.update()
                last_progress = time.monotonic()
                if self.learner.updates - self.published_step >= config.publish_updates:
                    self.publish()
                self.save_model()
                if self.learner.updates - self.last_full >= self.system.checkpoint_updates:
                    self._drain()
                    self.checkpoint()
            else:
                self.interval_data_wait_seconds += time.monotonic() - started
            if throttled:
                self.interval_actor_throttle_seconds += time.monotonic() - started
            if before != (self.games, self.learner.updates):
                last_progress = time.monotonic()
            elif time.monotonic() - last_progress > self.system.timeout:
                raise TimeoutError("asynchronous training made no game or learner progress")
        self._drain()

    def run(self):
        old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop_requested", True))
        watcher = None
        try:
            if self.system.evaluations:
                from .live import start_watcher

                watcher = start_watcher(self.directory, self.system)
            with ThreadPoolExecutor(max_workers=1) as self.batch_pool:
                if self.last_full == 0:
                    self.checkpoint()
                if self.experiment.schedule == "phased":
                    self.phased()
                else:
                    self.asynchronous()
                self.save_model(force=True)
                self.checkpoint()
                result = self.progress()
                if watcher is not None:
                    from .live import evaluation_status

                    started = time.monotonic()
                    watcher[1].set()
                    watcher[0].join()
                    result["evaluation_wait_seconds"] = time.monotonic() - started
                    if watcher[0].exitcode:
                        atomic_json(
                            self.directory / "evaluations/status.json",
                            {
                                "status": "failed",
                                "error": f"evaluator exited with status {watcher[0].exitcode}",
                                "time": time.time(),
                            },
                        )
                    result["evaluations"] = evaluation_status(self.directory, self.system)
                self.journal.write(
                    "finished",
                    **result,
                    reason="stopped" if self.stop_requested else "update_budget",
                )
                return result
        except BaseException as error:
            self.journal.write("failed", error=str(error), **self.progress())
            raise
        finally:
            if watcher is not None and watcher[0].is_alive():
                watcher[0].terminate()
                watcher[0].join()
            signal.signal(signal.SIGINT, old_handler)
            if self.pool:
                self.pool.close()
            self.group.close()
            self.journal.close()


def train(
    experiment,
    system,
    directory,
    resume=None,
    components=("representation", "dynamics", "prediction"),
):
    return TrainingRun(experiment, system, directory, resume, components).run()
