"""Paired Go matches against the retained, unmodified MiniZero executable."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import select
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from zero_systems.env import make_environment
from zero_systems.evaluate import fingerprint, load_candidate
from zero_systems.search import Search
from zero_systems.storage import versions


class Engine:
    def __init__(self, command, environment, log):
        self.log = log.open("wb")
        self.process = subprocess.Popen(
            command,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
        )
        self.buffer = b""

    def ask(self, command):
        self.process.stdin.write((command + "\n").encode())
        self.process.stdin.flush()
        deadline = time.monotonic() + 120
        while b"\n\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError(f"MiniZero did not answer {command!r}")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f"MiniZero exited during {command!r}; see {self.log.name}")
            self.buffer += chunk
        reply, self.buffer = self.buffer.split(b"\n\n", 1)
        reply = reply.decode().strip()
        if not reply.startswith("="):
            raise RuntimeError(f"MiniZero rejected {command!r}: {reply}")
        return reply[1:].strip()

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()


def vertex(action, side):
    return (
        "pass"
        if action == side * side
        else "ABCDEFGHJKLMNOPQRST"[action % side] + str(action // side + 1)
    )


def action_id(text, side):
    text = text.upper()
    return (
        side * side
        if text == "PASS"
        else (int(text[1:]) - 1) * side + "ABCDEFGHJKLMNOPQRST".index(text[0])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "opponent", "binary", "baseline_config", "output"):
        parser.add_argument(name)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7301)
    args = parser.parse_args()
    if args.games <= 0 or args.games % 2 or args.batch <= 0 or args.batch % 2:
        raise ValueError("games and batch must be positive even numbers")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.with_suffix(".logs")
    logs.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    metadata = versions()
    model, config, progress = load_candidate(args.candidate, args.device)
    if config.environment != "go" or config.history != 8 or len(config.komi) != 1:
        raise ValueError("native MiniZero requires fixed-komi Go with eight history states")
    if config.max_moves != 2 * config.board_size**2 + 1:
        raise ValueError("the move limit must match MiniZero")
    config = dataclasses.replace(config, simulations=args.simulations, search="gumbel")
    search = Search(model, config, args.device)
    side = config.board_size
    overrides = {
        "nn_file_name": str(Path(args.opponent).resolve()),
        "program_auto_seed": "false",
        "program_seed": args.seed,
        "program_quiet": "true",
        "env_board_size": side,
        "env_go_komi": config.komi[0],
        "env_go_ko_rule": "positional",
        "actor_num_simulation": args.simulations,
        "actor_mcts_think_batch_size": 1,
        "actor_mcts_think_time_limit": 0,
        "actor_select_action_by_count": "true",
        "actor_select_action_by_softmax_count": "false",
        "actor_use_gumbel": "true",
        "actor_use_gumbel_noise": "false",
        "actor_use_dirichlet_noise": "false",
        "actor_use_random_rotation_features": "false",
        "zero_disable_resign_ratio": 1,
    }
    command = [
        str(Path(args.binary).resolve()),
        "-mode",
        "console",
        "-conf_file",
        str(Path(args.baseline_config).resolve()),
        "-conf_str",
        ":".join(f"{key}={value}" for key, value in overrides.items()),
    ]
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="GPU-" + str(torch.cuda.get_device_properties(args.device).uuid),
    )
    engines = []
    scores, lengths = [], []
    started = time.monotonic()
    try:
        for row in range(min(args.batch, args.games)):
            engines.append(Engine(command, environment, logs / f"engine-{row}.log"))
        with ThreadPoolExecutor(max_workers=len(engines)) as executor:
            for start in range(0, args.games, args.batch):
                ids = list(range(start, min(args.games, start + args.batch)))
                environments = [make_environment(config, args.seed + i // 2) for i in ids]
                rngs = [
                    np.random.default_rng(np.random.SeedSequence([args.seed, i // 2])) for i in ids
                ]
                moves = [0] * len(ids)
                try:
                    list(
                        executor.map(lambda engine: engine.ask("clear_board"), engines[: len(ids)])
                    )
                    for row, env in enumerate(environments):
                        for _ in range(2):
                            position = env.position()
                            action = int(rngs[row].choice(np.flatnonzero(position.legal[:-1])))
                            engines[row].ask(f"play {'BW'[position.player]} {vertex(action, side)}")
                            env.step(action)
                            moves[row] += 1
                    active = list(range(len(ids)))
                    while active:
                        positions = {row: environments[row].position() for row in active}
                        ours = [row for row in active if positions[row].player == ids[row] % 2]
                        theirs = [row for row in active if row not in ours]
                        pending = {
                            row: executor.submit(
                                engines[row].ask, f"genmove {'BW'[positions[row].player]}"
                            )
                            for row in theirs
                        }
                        actions = {}
                        if ours:
                            result = search(
                                [positions[row] for row in ours],
                                [rngs[row] for row in ours],
                                explore=False,
                            )
                            for row, action in zip(ours, result.actions, strict=True):
                                actions[row] = int(action)
                                engines[row].ask(
                                    f"play {'BW'[positions[row].player]} {vertex(action, side)}"
                                )
                        for row, future in pending.items():
                            actions[row] = action_id(future.result(), side)
                        for row in active:
                            if (
                                not 0 <= actions[row] <= side * side
                                or not positions[row].legal[actions[row]]
                            ):
                                raise RuntimeError(
                                    f"native and OpenSpiel legality disagree in game {ids[row]}"
                                )
                            environments[row].step(actions[row])
                            moves[row] += 1
                        active = [row for row in active if not environments[row].terminated]
                    for row, env in enumerate(environments):
                        native_outcome = float(engines[row].ask("final_score"))
                        if native_outcome != env.outcome[0]:
                            raise RuntimeError(
                                f"native and OpenSpiel outcomes disagree in game {ids[row]}"
                            )
                        scores.append(float((env.outcome[ids[row] % 2] + 1) / 2))
                        lengths.append(moves[row])
                    print(
                        json.dumps(dict(completed=len(scores), score=float(np.mean(scores)))),
                        flush=True,
                    )
                finally:
                    for env in environments:
                        env.close()
    finally:
        for engine in engines:
            engine.close()
    values = np.asarray(scores)
    pairs = values.reshape(-1, 2).mean(1)
    bootstrap = np.random.default_rng(args.seed).choice(pairs, (4000, len(pairs))).mean(1)
    result = dict(
        protocol="Zero Systems versus native MiniZero; paired openings; no resignation",
        candidate={**fingerprint(args.candidate), "progress": progress},
        opponent=fingerprint(args.opponent),
        binary=fingerprint(args.binary),
        baseline_config=fingerprint(args.baseline_config),
        native_overrides=overrides,
        experiment=dataclasses.asdict(config),
        versions=metadata,
        device=args.device,
        games=args.games,
        seed=args.seed,
        opening_moves=2,
        wins=int(sum(values == 1)),
        draws=int(sum(values == 0.5)),
        losses=int(sum(values == 0)),
        score=float(values.mean()),
        paired_bootstrap_95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        pair_scores=pairs.tolist(),
        mean_moves=float(np.mean(lengths)),
        seconds=time.monotonic() - started,
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
