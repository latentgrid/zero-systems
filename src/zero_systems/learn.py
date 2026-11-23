"""A visible unrolled loss and one optimizer update, including uneven DDP shards."""

from __future__ import annotations

import contextlib

import torch
from torch import nn

from .config import Experiment, System, accumulation_passes
from .data import Batch


def scaled_gradient(x: torch.Tensor, scale: float):
    return x.detach() + scale * (x - x.detach())


class Objective(nn.Module):
    def __init__(self, model: nn.Module, config: Experiment):
        super().__init__()
        self.model, self.config = model, config

    def forward(self, batch: Batch):
        state, logits, value = self.model.initial_inference(batch.observation, batch.context)
        policy_loss = -(batch.policy[:, 0] * logits.float().log_softmax(-1)).sum(-1)
        value_loss = (value.float() - batch.value[:, 0]).square()
        policy_total = policy_loss * batch.valid[:, 0]
        value_total = self.config.value_weight * value_loss * batch.valid[:, 0]
        reward_total = torch.zeros_like(policy_total)
        for k in range(self.config.unroll):
            state, logits, value, reward = self.model.recurrent_inference(
                state, batch.actions[:, k]
            )
            # Only the next dynamics call receives the reduced gradient.
            # Current policy/value/reward predictions retain their full gradient.
            state = scaled_gradient(state, 0.5)
            policy_loss = -(batch.policy[:, k + 1] * logits.float().log_softmax(-1)).sum(-1)
            value_loss = (value.float() - batch.value[:, k + 1]).square()
            reward_loss = (reward.float() - batch.reward[:, k + 1]).square()
            valid = batch.valid[:, k + 1] / self.config.unroll
            policy_total = policy_total + policy_loss * valid
            value_total = value_total + self.config.value_weight * value_loss * valid
            reward_total = (
                reward_total
                + self.config.reward_weight
                * reward_loss
                * batch.reward_valid[:, k + 1]
                / self.config.unroll
            )
        return torch.stack((policy_total, value_total, reward_total), dim=1)


def autocast(device, precision):
    device = torch.device(device)
    if precision == "float32":
        return contextlib.nullcontext()
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError(f"bfloat16 is unsupported on {device}")
    return torch.autocast("cuda", dtype=torch.bfloat16)


def shard_bounds(batch_size: int, rank: int, world: int) -> tuple[int, int]:
    return batch_size * rank // world, batch_size * (rank + 1) // world


class Learner:
    def __init__(self, model, experiment: Experiment, system: System, device, *, distributed=False):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.experiment, self.system = experiment, system
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=experiment.learning_rate,
            momentum=experiment.momentum,
            weight_decay=experiment.weight_decay,
            fused=self.device.type == "cuda",
        )
        self.objective = Objective(self.model, experiment)
        if system.compile:
            self.objective = torch.compile(self.objective)
        self.world = torch.distributed.get_world_size() if distributed else 1
        self.rank = torch.distributed.get_rank() if distributed else 0
        if distributed:
            self.objective = nn.parallel.DistributedDataParallel(
                self.objective,
                device_ids=[self.device.index],
                bucket_cap_mb=system.ddp_bucket_mb,
                gradient_as_bucket_view=True,
            )
        self.updates = 0
        self.samples = 0

    def step(self, batch: Batch) -> dict[str, float]:
        global_size = len(batch)
        lo, hi = shard_bounds(global_size, self.rank, self.world)
        passes = accumulation_passes(global_size, self.world, self.system.microbatch)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros(3, device=self.device)
        for part in range(passes):
            start = lo + (hi - lo) * part // passes
            stop = lo + (hi - lo) * (part + 1) // passes
            local = batch.part(start, stop).to(self.device)
            sync = (
                self.objective.no_sync()
                if self.world > 1 and part < passes - 1
                else contextlib.nullcontext()
            )
            with sync, autocast(self.device, self.system.precision):
                losses = self.objective(local)
                loss = losses.sum() * self.world / global_size
                loss.backward()
            loss_sum += losses.detach().float().sum(0)
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.experiment.grad_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        if self.world > 1:
            torch.distributed.all_reduce(loss_sum)
        self.updates += 1
        self.samples += global_size
        return {
            "loss": float(loss_sum.sum() / global_size),
            "policy_loss": float(loss_sum[0] / global_size),
            "value_loss": float(loss_sum[1] / global_size),
            "reward_loss": float(loss_sum[2] / global_size),
            "grad_norm": float(grad_norm),
            "updates": self.updates,
            "learner_samples": self.samples,
        }
