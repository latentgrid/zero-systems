"""Batched MuZero search; the simulator is never used below a root.

Tree values are always from the player at that node. The inference boundary
converts model value conventions once, and backups change perspective once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numba
import numpy as np
import torch

from .config import Experiment
from .learn import autocast


@dataclass
class SearchResult:
    actions: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    visits: np.ndarray
    action_value: np.ndarray


def halving_visits(actions: int, simulations: int) -> np.ndarray:
    """Visit levels select the surviving half without a separate survivor set."""
    if actions <= 1:
        return np.arange(simulations, dtype=np.int32)
    levels = np.zeros(actions, np.int32)
    rounds = math.ceil(math.log2(actions))
    remaining = actions
    schedule = []
    while len(schedule) < simulations:
        repetitions = max(1, simulations // (rounds * remaining))
        for _ in range(repetitions):
            schedule.extend(levels[:remaining].tolist())
            levels[:remaining] += 1
        remaining = max(2, remaining // 2)
    return np.asarray(schedule[:simulations], np.int32)


@numba.njit(cache=True)
def _edge_values(children, visits, totals, raw, rewards, prior, parent, discount, players, use_raw):
    actions = prior.shape[1]
    counts = np.zeros(actions, np.int32)
    q = np.zeros(actions, np.float32)
    mass = 0.0
    weighted = 0.0
    total = 0
    sign = -1.0 if players == 2 else 1.0
    for action in range(actions):
        child = children[parent, action]
        if child >= 0:
            counts[action] = visits[child]
            if visits[child] > 0:
                estimate = raw[child] if use_raw else totals[child] / visits[child]
                q[action] = rewards[child] + discount * sign * estimate
                mass += prior[parent, action]
                weighted += prior[parent, action] * q[action]
                total += counts[action]
    mixed = raw[parent]
    if mass > 0:
        mixed = (mixed + total * weighted / mass) / (total + 1)
    for action in range(actions):
        if counts[action] == 0:
            q[action] = mixed
    return counts, q


@numba.njit(cache=True)
def _improved(prior, q, counts, scale, rescale):
    result = np.empty(len(prior), np.float32)
    low, high = 1e30, -1e30
    for a in range(len(prior)):
        if prior[a] > 0:
            low, high = min(low, q[a]), max(high, q[a])
    maximum = -1e30
    multiplier = (50 + counts.max()) * scale
    for a in range(len(prior)):
        value = (q[a] - low) / max(high - low, 1e-8) if rescale else q[a]
        result[a] = math.log(prior[a]) + multiplier * value if prior[a] > 0 else -1e30
        maximum = max(maximum, result[a])
    return result, maximum


@numba.njit(cache=True, nogil=True)
def _select(
    children,
    prior,
    visits,
    totals,
    raw,
    rewards,
    paths,
    noise,
    target_visits,
    discount,
    players,
    gumbel,
    q_scale,
    rescale,
    base,
    init,
    low,
    high,
):
    batch, _, actions = children.shape
    parents = np.empty(batch, np.int32)
    moves = np.empty(batch, np.int32)
    depths = np.empty(batch, np.int32)
    for row in range(batch):
        node = 0
        depth = 0
        while True:
            paths[row, depth] = node
            depth += 1
            counts, q = _edge_values(
                children[row],
                visits[row],
                totals[row],
                raw[row],
                rewards[row],
                prior[row],
                node,
                discount,
                players,
                False,
            )
            best, best_score = 0, -1e30
            if gumbel:
                logits, maximum = _improved(prior[row, node], q, counts, q_scale, rescale)
                normalizer = 0.0
                for a in range(actions):
                    normalizer += math.exp(logits[a] - maximum)
                for a in range(actions):
                    if prior[row, node, a] <= 0:
                        continue
                    if node == 0:
                        score = (
                            logits[a] + noise[row, a] if counts[a] == target_visits[row] else -1e30
                        )
                    else:
                        score = math.exp(logits[a] - maximum) / normalizer - counts[a] / (
                            1.0 + counts.sum()
                        )
                    if score > best_score:
                        best, best_score = a, score
            else:
                coefficient = (math.log((visits[row, node] + base + 1) / base) + init) * math.sqrt(
                    visits[row, node]
                )
                for a in range(actions):
                    if prior[row, node, a] <= 0:
                        continue
                    value = q[a] if counts[a] else 0.0
                    if rescale and counts[a]:
                        value = (value - low[row]) / max(high[row] - low[row], 1e-8)
                    score = value + coefficient * prior[row, node, a] / (counts[a] + 1)
                    if score > best_score:
                        best, best_score = a, score
            child = children[row, node, best]
            if child < 0:
                parents[row], moves[row], depths[row] = node, best, depth
                break
            node = child
    return parents, moves, depths


@numba.njit(cache=True, nogil=True)
def _backup(visits, totals, rewards, raw, paths, depths, new_node, discount, players, low, high):
    sign = -1.0 if players == 2 else 1.0
    for row in range(visits.shape[0]):
        value = raw[row, new_node]
        visits[row, new_node] = 1
        totals[row, new_node] = value
        child = new_node
        for step in range(depths[row] - 1, -1, -1):
            value = rewards[row, child] + discount * sign * value
            node = paths[row, step]
            visits[row, node] += 1
            totals[row, node] += value
            edge_mean = (
                rewards[row, child] + discount * sign * totals[row, child] / visits[row, child]
            )
            low[row] = min(low[row], edge_mean)
            high[row] = max(high[row], edge_mean)
            child = node


class Search:
    def __init__(
        self,
        model,
        config: Experiment,
        device="cpu",
        precision="float32",
        compile=False,
        gpu_search=True,
    ):
        self.model, self.config = model, config
        self.device, self.precision = torch.device(device), precision
        self._initial = model.initial_inference
        self._recurrent = model.recurrent_inference
        self.compiled = compile
        if compile:
            self._initial = torch.compile(self._initial, mode="reduce-overhead")
            self._recurrent = torch.compile(self._recurrent, mode="reduce-overhead")
        self.gpu_search = gpu_search

    def _infer(self, function, *inputs):
        count = len(inputs[0])
        if self.compiled:
            # Shrinking cohorts reuse power-of-two CUDA graphs. Padding affects
            # inference only; dummy rows never enter trees, RNG streams or replay.
            padded = 1 << (count - 1).bit_length()
            inputs = tuple(
                torch.cat((x, x.new_zeros((padded - count, *x.shape[1:]))))
                if padded != count
                else x
                for x in inputs
            )
            if self.device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
        return tuple(x[:count] for x in function(*inputs))

    def initial(self, observation, context):
        return self._infer(self._initial, observation, context)

    def recurrent(self, latent, actions):
        return self._infer(self._recurrent, latent, actions)

    @torch.inference_mode()
    def __call__(self, positions, rngs, *, explore=True, moves=None, noise=None):
        self.model.eval()
        config = self.config
        batch = len(positions)
        if batch == 0:
            raise ValueError("empty search batch")
        masks = np.stack([p.legal for p in positions])
        if not masks.any(1).all():
            raise ValueError("nonterminal root has no legal actions")
        actions = masks.shape[1]
        players = self.model.task.players
        root_players = np.asarray([p.player for p in positions], np.int32)
        observation = torch.as_tensor(
            np.stack([p.observation for p in positions]), device=self.device
        )
        context = torch.as_tensor(np.stack([p.context for p in positions]), device=self.device)
        with autocast(self.device, self.precision):
            state, logits, value = self.initial(observation, context)
        shape = (batch, config.simulations + 1)
        latent = torch.empty((*shape, *state.shape[1:]), dtype=state.dtype, device=self.device)
        latent[:, 0].copy_(state)
        prior = np.zeros((*shape, actions), np.float32)
        root_logits = logits.float().cpu().numpy()
        if not np.isfinite(root_logits).all():
            raise FloatingPointError("non-finite root policy logits")
        root_logits = np.where(masks, root_logits, -np.inf)
        probability = np.exp(root_logits - root_logits.max(1, keepdims=True))
        probability = np.where(masks, np.maximum(probability, 1e-30), 0)
        probability /= probability.sum(1, keepdims=True)
        if explore and config.search == "puct":
            for row, rng in enumerate(rngs):
                indices = np.flatnonzero(masks[row])
                fraction = config.dirichlet_fraction
                probability[row, indices] = (1 - fraction) * probability[
                    row, indices
                ] + fraction * rng.dirichlet(np.full(len(indices), config.dirichlet_alpha))
        prior[:, 0] = probability
        if noise is None:
            noise = np.zeros_like(probability)
            if explore and config.search == "gumbel":
                for row, rng in enumerate(rngs):
                    noise[row] = rng.gumbel(size=actions) * config.gumbel_scale
        children = np.full((*shape, actions), -1, np.int32)
        visits = np.zeros(shape, np.int32)
        totals = np.zeros(shape, np.float32)
        raw = np.zeros(shape, np.float32)
        rewards = np.zeros(shape, np.float32)
        paths = np.empty(shape, np.int32)
        node_players = np.zeros(shape, np.int32)
        node_players[:, 0] = root_players
        raw[:, 0] = value.float().cpu().numpy()
        if config.value_perspective == "player0":
            raw[:, 0] *= 1 - 2 * root_players
        visits[:, 0], totals[:, 0] = 1, raw[:, 0]
        schedules = np.stack(
            [
                halving_visits(min(config.considered_actions, int(mask.sum())), config.simulations)
                for mask in masks
            ]
        )
        low = np.full(batch, np.inf, np.float32)
        high = np.full(batch, -np.inf, np.float32)
        rows = np.arange(batch)
        gpu = None
        if self.device.type == "cuda" and self.gpu_search:
            from .search_gpu import GPUTree

            gpu = GPUTree(prior[:, 0], raw[:, 0], noise, schedules, config, players, self.device)
        for simulation in range(config.simulations):
            node = simulation + 1
            if gpu is None:
                parent, action, depth = _select(
                    children,
                    prior,
                    visits,
                    totals,
                    raw,
                    rewards,
                    paths,
                    noise,
                    schedules[:, simulation],
                    config.discount,
                    players,
                    config.search == "gumbel",
                    config.q_scale,
                    config.rescale_q,
                    config.puct_base,
                    config.puct_init,
                    low,
                    high,
                )
                parent_device = torch.as_tensor(parent, device=self.device)
                action_device = torch.as_tensor(action, device=self.device)
                node_players[:, node] = (node_players[rows, parent] + 1) % players
                player_device = torch.as_tensor(node_players[:, node], device=self.device)
            else:
                parent_device, action_device, player_device = gpu.select(simulation, root_players)
            inputs = latent[torch.arange(batch, device=self.device), parent_device.long()]
            with autocast(self.device, self.precision):
                state, logits, value, reward = self.recurrent(inputs, action_device.long())
            latent[:, node].copy_(state)
            if config.value_perspective == "player0":
                value = value * (1 - 2 * player_device)
                parent_player = (player_device + players - 1) % players
                reward = reward * (1 - 2 * parent_player)
            if gpu is not None:
                gpu.expand(
                    node, logits.float().softmax(-1).clamp_min(1e-30), value.float(), reward.float()
                )
            else:
                children[rows, parent, action] = node
                prior[:, node] = logits.float().softmax(-1).clamp_min(1e-30).cpu().numpy()
                raw[:, node] = value.float().cpu().numpy()
                rewards[:, node] = reward.float().cpu().numpy()
                _backup(
                    visits,
                    totals,
                    rewards,
                    raw,
                    paths,
                    depth,
                    node,
                    config.discount,
                    players,
                    low,
                    high,
                )
        if gpu is not None:
            children, prior, visits, totals, raw, rewards = gpu.cpu_arrays()
        selected = np.empty(batch, np.int64)
        selected_values = np.empty(batch, np.float32)
        target = np.empty_like(probability)
        root_counts = np.empty((batch, actions), np.int32)
        for row, rng in enumerate(rngs):
            counts, mean_q = _edge_values(
                children[row],
                visits[row],
                totals[row],
                raw[row],
                rewards[row],
                prior[row],
                0,
                config.discount,
                players,
                False,
            )
            root_counts[row] = counts
            if config.search == "gumbel":
                _, policy_q = _edge_values(
                    children[row],
                    visits[row],
                    totals[row],
                    raw[row],
                    rewards[row],
                    prior[row],
                    0,
                    config.discount,
                    players,
                    True,
                )
                improved, maximum = _improved(
                    prior[row, 0], policy_q, counts, config.q_scale, config.rescale_q
                )
                weights = np.exp(improved - maximum)
                target[row] = weights / weights.sum()
            else:
                target[row] = counts / counts.sum()
                selected[row] = counts.argmax()
            if explore and (moves is None or moves[row] < config.temperature_moves):
                weights = (counts.astype(np.float64) / counts.max()) ** (1 / config.temperature)
                if config.search == "gumbel":
                    reference = np.where(counts == counts.max(), prior[row, 0], -np.inf).argmax()
                    weights[mean_q < mean_q[reference] - config.action_threshold] = 0
                selected[row] = rng.choice(actions, p=weights / weights.sum())
            elif config.search == "gumbel":
                # Greedy decisions use backed-up means, even when the
                # MiniZero-compatible policy target uses initial child values.
                decision, _ = _improved(
                    prior[row, 0], mean_q, counts, config.q_scale, config.rescale_q
                )
                selected[row] = np.where(
                    counts == counts.max(), noise[row] + decision, -np.inf
                ).argmax()
            selected_values[row] = mean_q[selected[row]]
        # The prior evaluation seeds selection, but the returned search value
        # averages simulation backups only (also used for targets/resignation).
        root_value = (totals[:, 0] - raw[:, 0]) / (visits[:, 0] - 1)
        if (
            not np.isfinite(root_value).all()
            or not np.isfinite(selected_values).all()
            or not np.isfinite(target).all()
        ):
            raise FloatingPointError("non-finite search predictions")
        return SearchResult(selected, target, root_value, root_counts, selected_values)
