"""Finite actor cohorts with immediate completed-game delivery."""

from __future__ import annotations

import numpy as np
import torch

from .config import Experiment
from .data import Trajectory, game_seed
from .env import make_environment
from .learn import autocast


def play_games(
    search, config: Experiment, game_ids, model_step: int, *, emit=None, cancelled=lambda: False
):
    envs, records, rngs = [], [], []
    try:
        for game_id in game_ids:
            seed = game_seed(config.seed, game_id)
            envs.append(make_environment(config, seed))
            rng = np.random.default_rng(game_seed(config.seed, game_id, 1))
            rngs.append(rng)
            records.append(
                {
                    "id": game_id,
                    "seed": seed,
                    "obs": [],
                    "context": [],
                    "players": [],
                    "actions": [],
                    "rewards": [],
                    "policies": [],
                    "values": [],
                    "resign": rng.random() >= config.no_resign_fraction,
                    "would_resign": -1,
                }
            )
        active = list(range(len(envs)))
        completed = []
        while active:
            if cancelled():
                raise InterruptedError("self-play cancelled")
            positions = [envs[i].position() for i in active]
            result = search(
                positions,
                [rngs[i] for i in active],
                moves=[len(records[i]["actions"]) for i in active],
            )
            remaining = []
            for row, index in enumerate(active):
                env, record, position = envs[index], records[index], positions[row]
                would_resign = (
                    hasattr(env, "resign")
                    and model_step >= config.resign_after
                    and result.value[row] < config.resign_threshold
                    and result.action_value[row] < config.resign_threshold
                    and len(record["actions"]) > 0
                )
                if would_resign and record["would_resign"] < 0:
                    record["would_resign"] = position.player
                resign = would_resign and record["resign"]
                if resign:
                    env.resign(position.player)
                else:
                    record["obs"].append(position.observation)
                    record["context"].append(position.context)
                    record["players"].append(position.player)
                    record["actions"].append(result.actions[row])
                    record["policies"].append(result.policy[row])
                    sign = 1 - 2 * position.player if config.value_perspective == "player0" else 1
                    record["values"].append(result.value[row] * sign)
                    record["rewards"].append(env.step(int(result.actions[row])))
                if not env.terminated and not env.truncated:
                    remaining.append(index)
                    continue
                bootstrap, bootstrap_player = 0.0, 0
                if env.truncated:
                    final = env.position()
                    with torch.inference_mode(), autocast(search.device, search.precision):
                        _, _, estimate = search.initial(
                            torch.as_tensor(final.observation[None], device=search.device),
                            torch.as_tensor(final.context[None], device=search.device),
                        )
                    bootstrap = float(estimate[0])
                    bootstrap_player = 0 if config.value_perspective == "player0" else final.player
                game = Trajectory(
                    record["id"],
                    record["seed"],
                    np.stack(record["obs"]),
                    np.stack(record["context"]),
                    np.asarray(record["players"], np.int32),
                    np.asarray(record["actions"], np.int32),
                    np.asarray(record["rewards"], np.float32),
                    np.asarray(record["policies"], np.float32),
                    np.asarray(record["values"], np.float32),
                    model_step,
                    env.terminated,
                    env.truncated,
                    bootstrap,
                    bootstrap_player,
                    env.outcome,
                    resigned_player=position.player if resign else -1,
                    resignation_test_player=record["would_resign"] if not record["resign"] else -1,
                )
                game.validate(env.task)
                if emit:
                    emit(game)
                else:
                    completed.append(game)
            active = remaining
        return completed
    finally:
        for env in envs:
            env.close()
