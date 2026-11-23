"""Explicit commands for training, remote actors, evaluation and measurement."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from .config import Experiment, System, load


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zero-systems")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="start or resume training")
    training.add_argument("experiment")
    training.add_argument("system")
    training.add_argument("directory")
    start_from = training.add_mutually_exclusive_group()
    start_from.add_argument("--resume")
    start_from.add_argument("--initialize")
    training.add_argument("--updates", type=int)
    training.add_argument("--max-positions", type=int)
    training.add_argument(
        "--components", nargs="+", choices=("representation", "dynamics", "prediction")
    )
    actors = commands.add_parser("actors", help="connect self-play workers to a learner")
    actors.add_argument("address")
    actors.add_argument("system")
    conversion = commands.add_parser("convert", help="convert external weights for evaluation")
    conversion.add_argument("weights")
    conversion.add_argument("experiment")
    conversion.add_argument("output")
    conversion.add_argument("--updates", type=int, required=True)
    evaluation = commands.add_parser("evaluate", help="paired matches with common search")
    evaluation.add_argument("candidate")
    evaluation.add_argument("opponent")
    evaluation.add_argument("output")
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--games", type=int, default=200)
    evaluation.add_argument("--batch", type=int, default=64)
    evaluation.add_argument("--simulations", type=int, default=16)
    evaluation.add_argument("--seed", type=int, default=303)
    evaluation.add_argument("--opening-moves", type=int, default=2)
    evaluation.add_argument("--search", choices=("puct", "gumbel"))
    evaluation.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    episodes = commands.add_parser("evaluate-episodes", help="fixed-seed single-player returns")
    episodes.add_argument("candidate")
    episodes.add_argument("output")
    episodes.add_argument("--episodes", type=int, default=30)
    episodes.add_argument("--batch", type=int, default=32)
    episodes.add_argument("--simulations", type=int, default=16)
    episodes.add_argument("--seed", type=int, default=7301)
    episodes.add_argument("--device", default="cpu")
    episodes.add_argument("--search", choices=("puct", "gumbel"))
    episodes.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    watcher = commands.add_parser("watch", help="evaluate new portable checkpoints")
    watcher.add_argument("directory")
    watcher.add_argument("system")
    watcher.add_argument("--once", action="store_true")
    benchmark = commands.add_parser(
        "benchmark", help="fixed-input warmed search and learner timings"
    )
    benchmark.add_argument("experiment")
    benchmark.add_argument("system")
    benchmark.add_argument("output")
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command in {"evaluate", "evaluate-episodes"}:
        output = Path(args.output)
        checkpoints = [Path(args.candidate)]
        if args.command == "evaluate" and args.opponent != "random":
            checkpoints.append(Path(args.opponent))
        if output.exists() and any(p.exists() and output.samefile(p) for p in checkpoints):
            parser.error("evaluation output must be a different file from the input checkpoints")

    if args.command == "train":
        from .run import train

        experiment = load(args.experiment, Experiment)
        overrides = {
            key: getattr(args, key)
            for key in ("updates", "initialize", "max_positions")
            if getattr(args, key) is not None
        }
        experiment = dataclasses.replace(experiment, **overrides)
        if args.components and (args.resume or not experiment.initialize):
            parser.error("--components requires initialization for a new run")
        result = train(
            experiment,
            load(args.system, System),
            args.directory,
            args.resume,
            tuple(args.components or ("representation", "dynamics", "prediction")),
        )
    elif args.command == "actors":
        from .workers import actor_node

        actor_node(args.address, load(args.system, System))
        return
    elif args.command == "convert":
        from .reference import convert_checkpoint

        convert_checkpoint(
            args.weights, load(args.experiment, Experiment), args.output, args.updates
        )
        print(args.output)
        return
    elif args.command == "evaluate":
        from .evaluate import evaluate

        result = evaluate(
            args.candidate,
            args.opponent,
            args.device,
            args.games,
            args.batch,
            args.simulations,
            args.seed,
            args.opening_moves,
            search=args.search,
            precision=args.precision,
        )
    elif args.command == "evaluate-episodes":
        from .evaluate import evaluate_episodes

        result = evaluate_episodes(
            args.candidate,
            args.device,
            args.episodes,
            args.batch,
            args.simulations,
            args.seed,
            search=args.search,
            precision=args.precision,
        )
    elif args.command == "watch":
        from .live import watch

        watch(args.directory, load(args.system, System), once=args.once)
        return
    else:
        from .benchmark import benchmark

        result = benchmark(
            load(args.experiment, Experiment), load(args.system, System), args.repeats, args.warmup
        )
    if hasattr(args, "output"):
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
