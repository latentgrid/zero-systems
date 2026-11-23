"""Fresh Triton implementation of the same per-root tree operations as search.py.

One GPU program owns one tree. No atomics or virtual loss are necessary because
each simulation expands one leaf per independent root.
"""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def choose(
    CHILD,
    PRIOR,
    VISIT,
    TOTAL,
    RAW,
    REWARD,
    PATH,
    DEPTH,
    PARENT,
    MOVE,
    PLAYER,
    ROOT_PLAYER,
    NOISE,
    SCHEDULE,
    LOW,
    HIGH,
    A: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    SIM: tl.constexpr,
    GAMMA: tl.constexpr,
    SIGN: tl.constexpr,
    PLAYERS: tl.constexpr,
    GUMBEL: tl.constexpr,
    SCALE: tl.constexpr,
    RESCALE: tl.constexpr,
    BASE: tl.constexpr,
    INIT: tl.constexpr,
):
    row = tl.program_id(0)
    a = tl.arange(0, BLOCK)
    node = tl.full((), 0, tl.int32)
    depth = tl.full((), 0, tl.int32)
    searching = tl.full((), True, tl.int1)
    while searching:
        tl.store(PATH + row * N + depth, node)
        depth += 1
        edge = (row * N + node) * A + a
        child = tl.load(CHILD + edge, a < A, -1)
        prior = tl.load(PRIOR + edge, a < A, 0.0)
        address = row * N + tl.maximum(child, 0)
        count = tl.load(VISIT + address, (a < A) & (child >= 0), 0)
        visited = count > 0
        value = tl.load(TOTAL + address, (a < A) & (child >= 0), 0.0) / tl.maximum(count, 1)
        reward = tl.load(REWARD + address, (a < A) & (child >= 0), 0.0)
        q = reward + GAMMA * SIGN * value
        mass = tl.sum(tl.where(visited, prior, 0.0), 0)
        weighted = tl.sum(tl.where(visited, prior * q, 0.0), 0)
        total = tl.sum(count, 0)
        own_value = tl.load(RAW + row * N + node)
        mixed = tl.where(
            mass > 0,
            (own_value + total * weighted / tl.maximum(mass, 1e-30)) / (1 + total),
            own_value,
        )
        q = tl.where(visited, q, mixed)
        legal = (a < A) & (prior > 0)
        if GUMBEL:
            if RESCALE:
                q_low = tl.min(tl.where(legal, q, float("inf")), 0)
                q_high = tl.max(tl.where(legal, q, -float("inf")), 0)
                q = (q - q_low) / tl.maximum(q_high - q_low, 1e-8)
            logits = tl.log(tl.maximum(prior, 1e-30)) + (50 + tl.max(count, 0)) * SCALE * q
            logits = tl.where(legal, logits, -float("inf"))
            if node == 0:
                level = tl.load(SCHEDULE + row * (N - 1) + SIM)
                noise = tl.load(NOISE + row * A + a, a < A, 0.0)
                score = tl.where(legal & (count == level), logits + noise, -float("inf"))
            else:
                weights = tl.exp(logits - tl.max(logits, 0))
                score = tl.where(
                    legal, weights / tl.sum(weights, 0) - count / (1.0 + total), -float("inf")
                )
        else:
            parent_visits = tl.load(VISIT + row * N + node)
            coefficient = (tl.log((parent_visits + BASE + 1) / BASE) + INIT) * tl.sqrt(
                parent_visits.to(tl.float32)
            )
            if RESCALE:
                low = tl.load(LOW + row)
                high = tl.load(HIGH + row)
                q = (q - low) / tl.maximum(high - low, 1e-8)
            score = tl.where(
                legal, tl.where(visited, q, 0.0) + coefficient * prior / (count + 1), -float("inf")
            )
        best = tl.min(tl.where(score == tl.max(score, 0), a, 2147483647), 0)
        next_node = tl.load(CHILD + (row * N + node) * A + best)
        if next_node < 0:
            tl.store(PARENT + row, node)
            tl.store(MOVE + row, best)
            tl.store(DEPTH + row, depth)
            tl.store(PLAYER + row, (tl.load(ROOT_PLAYER + row) + depth) % PLAYERS)
            searching = False
        else:
            node = next_node


@triton.jit
def expand_and_backup(
    CHILD,
    PRIOR,
    VISIT,
    TOTAL,
    RAW,
    REWARD,
    PATH,
    DEPTH,
    PARENT,
    MOVE,
    LOW,
    HIGH,
    NEW_PRIOR,
    NEW_VALUE,
    NEW_REWARD,
    A: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    NODE: tl.constexpr,
    GAMMA: tl.constexpr,
    SIGN: tl.constexpr,
):
    row = tl.program_id(0)
    a = tl.arange(0, BLOCK)
    parent = tl.load(PARENT + row)
    action = tl.load(MOVE + row)
    tl.store(CHILD + (row * N + parent) * A + action, NODE)
    p = tl.load(NEW_PRIOR + row * A + a, a < A, 0.0)
    tl.store(PRIOR + (row * N + NODE) * A + a, p, a < A)
    value = tl.load(NEW_VALUE + row)
    tl.store(RAW + row * N + NODE, value)
    tl.store(TOTAL + row * N + NODE, value)
    tl.store(VISIT + row * N + NODE, 1)
    tl.store(REWARD + row * N + NODE, tl.load(NEW_REWARD + row))
    depth = tl.load(DEPTH + row)
    child = tl.full((), NODE, tl.int32)
    low = tl.load(LOW + row)
    high = tl.load(HIGH + row)
    while depth > 0:
        depth -= 1
        node = tl.load(PATH + row * N + depth)
        reward = tl.load(REWARD + row * N + child)
        value = reward + GAMMA * SIGN * value
        address = row * N + node
        tl.store(TOTAL + address, tl.load(TOTAL + address) + value)
        tl.store(VISIT + address, tl.load(VISIT + address) + 1)
        q = reward + GAMMA * SIGN * tl.load(TOTAL + row * N + child) / tl.load(
            VISIT + row * N + child
        )
        low = tl.minimum(low, q)
        high = tl.maximum(high, q)
        child = node
    tl.store(LOW + row, low)
    tl.store(HIGH + row, high)


class GPUTree:
    def __init__(self, prior, values, noise, schedules, config, players, device):
        self.config, self.players, self.device = config, players, device
        self.batch, self.actions = prior.shape
        self.nodes = config.simulations + 1
        shape = (self.batch, self.nodes)
        self.children = torch.full((*shape, self.actions), -1, dtype=torch.int32, device=device)
        self.prior = torch.zeros((*shape, self.actions), device=device)
        self.prior[:, 0] = torch.as_tensor(prior, device=device)
        self.visits = torch.zeros(shape, dtype=torch.int32, device=device)
        self.visits[:, 0] = 1
        self.total = torch.zeros(shape, device=device)
        self.total[:, 0] = torch.as_tensor(values, device=device)
        self.raw = self.total.clone()
        self.reward = torch.zeros(shape, device=device)
        self.paths = torch.empty(shape, dtype=torch.int32, device=device)
        self.depth = torch.empty(self.batch, dtype=torch.int32, device=device)
        self.parent = torch.empty_like(self.depth)
        self.move = torch.empty_like(self.depth)
        self.player = torch.empty_like(self.depth)
        self.root_player = None
        self.noise = torch.as_tensor(np.asarray(noise, np.float32), device=device)
        self.schedule = torch.as_tensor(schedules, device=device)
        self.low = torch.full((self.batch,), float("inf"), device=device)
        self.high = -self.low

    def select(self, simulation, root_players):
        if self.root_player is None:
            self.root_player = torch.as_tensor(root_players, device=self.device)
        c = self.config
        with torch.cuda.device(self.device):
            choose[(self.batch,)](
                self.children,
                self.prior,
                self.visits,
                self.total,
                self.raw,
                self.reward,
                self.paths,
                self.depth,
                self.parent,
                self.move,
                self.player,
                self.root_player,
                self.noise,
                self.schedule,
                self.low,
                self.high,
                self.actions,
                self.nodes,
                triton.next_power_of_2(self.actions),
                simulation,
                c.discount,
                -1.0 if self.players == 2 else 1.0,
                self.players,
                c.search == "gumbel",
                c.q_scale,
                c.rescale_q,
                c.puct_base,
                c.puct_init,
            )
        return self.parent, self.move, self.player

    def expand(self, node, prior, value, reward):
        with torch.cuda.device(self.device):
            expand_and_backup[(self.batch,)](
                self.children,
                self.prior,
                self.visits,
                self.total,
                self.raw,
                self.reward,
                self.paths,
                self.depth,
                self.parent,
                self.move,
                self.low,
                self.high,
                prior.contiguous(),
                value.contiguous(),
                reward.contiguous(),
                self.actions,
                self.nodes,
                triton.next_power_of_2(self.actions),
                node,
                self.config.discount,
                -1.0 if self.players == 2 else 1.0,
            )

    def cpu_arrays(self):
        return tuple(
            t.cpu().numpy()
            for t in (
                self.children[:, :1],
                self.prior[:, :1],
                self.visits,
                self.total,
                self.raw,
                self.reward,
            )
        )
