# Zero Systems

A readable MuZero research implementation in Python and PyTorch, with PUCT and
Gumbel search, local or remote self-play, and one-node single/multi-GPU learning.
The package is `zero_systems`; the command is `zero-systems`.

[ARCHITECTURE.md](ARCHITECTURE.md) explains the implementation.
[VALIDATION.md](VALIDATION.md) records the checked behavior, development
measurements, and remaining limits.

## Install and run

Python 3.12 is the locked development environment. Run commands from this
repository root:

```bash
uv sync --extra all --extra dev
uv run zero-systems train configs/cartpole.toml systems/cpu-monitored.toml runs/cartpole
uv run zero-systems evaluate-episodes runs/cartpole/models/step_00001000.pt runs/cartpole-final.json --episodes 30 --seed 17301
uv run tensorboard --logdir runs/cartpole/tensorboard
```

Experiment files describe the algorithm and environment; system files describe
hardware, checkpoint cadence, and monitoring.

| Experiment | Purpose |
| --- | --- |
| `configs/go9.toml` | Validated fixed-offset Go network, async 10 games/update, 60k development budget |
| `configs/go9-puct.toml` | Original-style PUCT development preset; not a paper reproduction |
| `configs/go9-reference.toml` | Original head for external weight conversion |
| `configs/cartpole.toml` / `cartpole-async.toml` | Phased / position-paced single-player training |
| `configs/frozenlake.toml` / `taxi.toml` | Scalar discrete observations and action masks |
| `configs/tictactoe.toml` / `connect-four.toml` | General two-player OpenSpiel with the MLP |

For one GPU, use `systems/single-gpu-bf16.toml`, or
`single-gpu-monitored.toml` for TensorBoard and the fixed MiniZero anchor:

```bash
uv run zero-systems train configs/go9.toml systems/single-gpu-monitored.toml runs/go9
```

The monitored Go preset expects the retained sibling-repository anchor at
`../Project-Data-Science/artifacts/baselines/minizero-selftrained/weight_iter_060000.pt`.
Set `opponent` to an existing absolute checkpoint path on another machine.
`systems/four-plus-two.toml` separates learner and actor GPUs on one node.
CUDA indices are PyTorch indices; confirm the machine's device mapping.
Compilation has a first-use cost, included in training elapsed time.

## Environment contract

- `go`: OpenSpiel rules with Go-specific relative history planes, komi context,
  symmetry augmentation, and resignation. Extra rule parameters go in
  `environment_kwargs`; use explicit `board_size`, `komi`, and `max_moves`
  fields for those parameters.
- `gym:ENVIRONMENT_ID`: fixed-size numeric Box, Discrete, MultiBinary,
  MultiDiscrete, Dict, or Tuple observations flattened with Gymnasium's standard
  utilities; Discrete actions, including nonzero starts. Use `network = "mlp"`
  and `augmentation = false`.
- `spiel:GAME_NAME`: deterministic, perfect-information, alternating one-player
  or two-player zero-sum games, using observation tensors and the MLP. Chance,
  simultaneous, imperfect-information, non-zero-sum multiplayer, and extra-turn
  semantics are rejected.

Stochastic Gymnasium environments are allowed; this remains deterministic
MuZero latent planning, without an explicit stochastic search model.
Constructors receive the `[environment_kwargs]` TOML table. Gym masks default
to `info["action_mask"]` when present. Set `action_mask_source = "observation"`
and `action_mask_key = "NAME"` for a required top-level Dict observation field;
that field is excluded from encoded observations. Masks must be binary, match
the action count, and provide a legal action at decision positions. Search and
replay use zero-based actions; the adapter translates environment action IDs.

Rewards retain their finite magnitude. Use `rescale_q = true` with PUCT for
unbounded returns, as in the CartPole preset. Terminal transitions retain their
rewards, and subsequent latent steps learn zero remaining return. Truncations
bootstrap from the final observation. The rollout manages individual
environments, without autoreset wrappers. Generic environments do not use Go's
outcome bounds, augmentation, or resignation. Checkpoints record encodings and
environment parameters, checked on initialization and evaluation.

## Scheduling and recovery

| Mode | Configuration | Meaning |
| --- | --- | --- |
| Phased | `schedule = "phased"`, `games_per_round`, `updates_per_round` | Collect complete games, then update; the final update round may be partial |
| Async games | `schedule = "async"`, `games_per_update` | Exact configured game/update budget |
| Async positions | `schedule = "async"`, `samples_per_position` | Learner sampling credit from cumulative accepted positions |

Only one explicit async ratio may be set. Old async files and checkpoints using
`games_per_round / updates_per_round` normalize to the equivalent game ratio.
Fractional game ratios must specify a whole final game budget.

Position mode allows the next update when `S + B <= r * P` after
`warmup_positions`: `P` counts unique accepted decision positions, `S` counts
consumed learner roots, and `B` is the **global** batch. Each root and its unroll
count once, regardless of GPU count or microbatching. Warmup earns credit;
eviction does not remove it. Retries, duplicates, and evaluations earn none.
`position_lead` bounds collection ahead of consumed credit. New assignments
pause at that lead; already assigned complete games may overshoot it. The
journal records overshoot, throttling, credit, and the actual ratio after
draining. Replay capacity and update/time stopping budgets are separate.
An optional `max_positions` (or `--max-positions`) stops new assignments at an
accepted-position budget. Assigned complete games still count, and learning
finishes the credit they earn. The final reason is `position_budget`, with less
than one batch of unused credit and explicit `position_budget_overshoot`.
Update, time and interrupt limits can stop earlier. A budget must cover warmup
and at least one batch. Increasing the budget on full resume keeps prior credit;
initializing a new stage starts every scheduling counter at zero.
Position-budget runs also save `models/final.pt` with the fully drained counters.
It is immutable during an attempt and archived before resuming that run.

```bash
uv run zero-systems train EXPERIMENT SYSTEM RUN --resume RUN/checkpoints/step_00001000.pt --updates 2000
```

Full checkpoints include replay, optimizer, RNGs, prepared batches, publication,
and scheduling counters, saved at accounted-work boundaries. Two full snapshots
are retained. Portable `models/step_XXXXXXXX.pt` snapshots remain available for
immutable evaluation. On rollback, later snapshots/results and old
JSONL/TensorBoard history move to `history/`; active history is rebuilt at the
resumed checkpoint. `System.max_seconds` and Ctrl-C stop at a recoverable
boundary. Full checkpoints are trusted local Python artifacts.

## Evaluation and TensorBoard

```bash
uv run zero-systems evaluate CANDIDATE.pt OPPONENT.pt results/match.json --games 200 --simulations 16 --search gumbel --device cuda:0
uv run zero-systems evaluate CANDIDATE.pt random results/random.json --games 200
uv run zero-systems evaluate-episodes CANDIDATE.pt results/returns.json --episodes 30 --seed 17301
uv run zero-systems watch RUN SYSTEM --once
```

Add `[[evaluations]]` tables to the system TOML. One optional evaluator process
executes all named jobs sequentially, loading immutable portable snapshots.
`mode` is `episodes`, `fixed`, or `self_lag`; `games` is the match or
episode count. Tables also specify `interval_updates`, `seed`, `simulations`,
`search`, `opening_moves`, `device`, `batch`, and `precision`. Fixed mode
uses `opponent = "random"` or a checkpoint path. Self-lag uses `lag_updates`
and skips explicitly if no older snapshot qualifies. Several fixed
configurations form a ladder. Promotion and Elo fitting are deferred.

The Go monitored preset uses 100 games every 6,000 updates and at completion.
The single-player preset uses 30 fixed-seed episodes every 200 updates and at
completion. Competitive matches pair seats and seeded openings and bootstrap
uncertainty over pairs; episode evaluation reports raw returns and episode-level
intervals. Exploration and resignation are disabled; both players use explicit
common search settings. Use separate seeds and larger final research budgets.

The watcher takes the latest eligible snapshot when behind, records skipped
snapshots, persists completed job identities for restart deduplication, and
surfaces failures in `evaluations/status.json`. Final evaluation can extend
command completion; `evaluation_wait_seconds` is separate from training time.
Standalone `watch` without `--once` follows a run until its final event. Only
one watcher may own a run. Native MiniZero corroboration remains in
`tools/native_arena.py`.

`metrics.jsonl` is authoritative. Set `tensorboard = true` for a buffered
mirror: progress, learning losses/entropy/KL/gradient norm, replay
age/credit/ratios, self-play returns/termination, throughput contributions, and
timing. Ordinary metrics use updates. Separate evaluator event directories also
plot strength/return against the evaluated checkpoint's positions, learner
samples, and elapsed training time. No per-move logging or metric-only GPU
synchronization is added. Self-play utility is not a strength metric.

## Multiple nodes and Docker

```bash
export ZERO_SYSTEMS_AUTHKEY="a-long-random-secret"
uv run zero-systems train configs/go9.toml systems/remote-learner.toml runs/remote
# On an actor host with the same source and auth key:
uv run zero-systems actors LEARNER_HOST:47631 systems/remote-actors.toml

docker build -t zero-systems:local .
docker run --rm --network host --gpus all --shm-size=2g -e ZERO_SYSTEMS_AUTHKEY zero-systems:local actors LEARNER_HOST:47631 systems/remote-actors.toml
```

Remote nodes receive finite, deduplicated assignments and immutable publications;
missing games are retried after disconnect. Authentication does not encrypt
traffic: use a trusted LAN or SSH tunnel. The physical test launcher
`tools/run_multinode.py` is specific to the development SSH/Docker host and
manages only its own container, tunnel, and MPS daemon. The production runtime
does not manage system-wide GPU services.

## Models and transfer experiments

The main path is `run.py -> workers.py / play.py -> search.py -> replay.py ->
learn.py`. Environment behavior stays in `env.py`; `evaluate.py` serves offline
and live evaluation, and `storage.py` owns snapshots and the journal. See [ARCHITECTURE.md](ARCHITECTURE.md) for algorithm contracts and checkpoint
compatibility.

`--initialize CHECKPOINT --components representation dynamics prediction`
copies selected compatible components into a fresh run. The default residual
`prediction_head = "dense"` retains the existing architecture and checkpoint
behavior: incompatible spatial head tensors reset, and adding komi context
zero-extends input weights.
For new board-size studies, set `prediction_head = "spatial"`. Shared point
logits use pooled global context; pass and value use pooled features plus board
height/width. Every learned shape is independent of board size.
`network = "transformer"` uses the same spatial heads with point tokens, fixed
2D sinusoidal coordinates, spatial action/pass encoding and learned dynamics.
It adds no dependency; set `attention_heads` to divide `channels` (also a multiple
of four). Dense value-bottleneck options do not apply to spatial heads.

Models are ordinary `nn.Module`s. The contract is
`initial_inference(observation, context) -> latent, policy_logits, value` and
`recurrent_inference(latent, action) -> latent, policy_logits, value, reward`.
Batch is the first axis, action IDs are zero-based, and search treats latent
axes as opaque. The named components are `representation` (h), `dynamics` (g),
and `prediction` (f). The board models retain Go's zero intermediate reward;
the MLP remains the general numeric-observation model.

Spatial initialization strictly copies **all** selected tensors, including
normalization buffers. It checks architecture, history/context encoding and value
semantics while allowing different board sizes and task komi. Missing or
mismatched selected tensors fail. Portable and full checkpoints retain the
source hash, source progress, selected components and copied/reset tensor names.
Optimizer, replay, RNG and stage counters start fresh; every parameter fine-tunes.
No freezing, optimizer/loss additions, reanalysis or curriculum runner is added.
Use `--components representation dynamics` to transfer both learned components
while starting the complete prediction component fresh.
Use `--components representation` for h-only or include `prediction` for full
transfer. The bounded study and its source-inclusive accounting live in the
sibling research repository's `experiments/09-board-transfer` directory.

```bash
uv run ruff check src/zero_systems tests tools
uv run pytest -q tests
uv run zero-systems benchmark configs/go9.toml systems/single-gpu-bf16.toml results/benchmark.json --warmup 3 --repeats 5
```

[VALIDATION.md](VALIDATION.md) describes the completed bounded checks.
Local `archive/` and `artifacts/` folders retain source snapshots and research
evidence outside Git. Source snapshots include their original dependency locks
and commands. The maintained package does not load archived implementations.

MIT. See `LICENSE`.
