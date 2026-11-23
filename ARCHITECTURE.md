# Architecture

Zero Systems implements discrete-action MuZero with an explicit training loop.
Experiment settings describe learning and the environment; system settings
describe hardware, output cadence, and evaluation. The primary path is:

```text
run.py -> workers.py / play.py -> search.py -> environment steps
  |                                  |
  +-> replay.py -> learn.py -> model.py+
  |
  +-> storage.py -> evaluate.py / live.py
```

## Learning and search

Representation encodes observations into a normalized latent state. Dynamics
predicts a new latent state and transition reward from the current latent and
action. Prediction produces policy logits and a scalar value. `model.py` keeps
the residual Go network and general MLP as ordinary PyTorch modules.

PUCT and Gumbel share inference, value conventions, replay, and losses. Root
legality comes from the environment; deeper transitions use learned dynamics.
CPU search is the readable reference for the optional GPU tree operations.
Each root owns its tree, so batched search requires no virtual loss or atomics.
Evaluation disables exploration and resignation.

Values use either player-zero or player-to-play perspective, with conversions
at inference boundaries. Two-player backups change perspective once per edge;
single-player rewards retain their sign. Generic values and rewards retain their
finite magnitudes. Go uses episodic outcomes, a bounded value head, and no learned
transition reward. The MLP has unbounded scalar value/reward heads.

Replay samples decision positions uniformly. One sampled root and its unroll
count as one learner sample. In transition-reward games, the terminal transition
keeps its reward; the terminal latent and subsequent unroll steps learn zero
remaining value, zero further rewards and a uniform policy. This prevents search
from adding unconstrained return after termination. Go instead retains its
separate outcome targets. Truncations bootstrap from the final observation and
do not train an absorbing tail. Go augmentation transforms
observations, policies, and unrolled actions together. Target returns are cached
until the immutable trajectory is evicted.

Root losses have unit weight; recurrent losses average over the unroll. Hidden
gradient scaling applies to the next dynamics call, preserving the current
prediction gradients. SGD, explicit clipping, and the present loss choices stay
small and visible in `learn.py`.

## Environments

`env.py` owns three thin adapter paths:

- Go: OpenSpiel rules, relative player/opponent history, explicit turn planes,
  positional superko, pass, komi context, and resignation diagnostics.
- Gymnasium: fixed numeric spaces flattened with standard utilities, Discrete
  actions translated to zero-based IDs, validated masks, raw rewards, and
  separate termination/truncation handling.
- OpenSpiel: deterministic, perfect-information, alternating one-player or
  two-player zero-sum games. Unsupported semantics fail explicitly.

Environment parameters and observation/action encodings accompany checkpoints.
Initialization and evaluation check their compatibility. Go-specific behavior
does not leak into the generic adapters. See `README.md` for configuration syntax.

## Scheduling and execution

The coordinator in `run.py` alone accepts game identities, mutates replay, and
advances training counters. Phased execution collects complete games and then
performs updates. Async execution interleaves the same operations with either
game-ratio or position-ratio credit.

For position mode, update eligibility is `S + B <= r * P` after warmup: `P` is
cumulative unique accepted positions, `S` is consumed learner roots, and `B` is
global batch size. Warmup earns credit; eviction, retries, and evaluation do not
alter it. Dispatch pauses at the lead limit while assigned complete games drain.
Final progress reports actual sampling ratio and any bounded lead overshoot.

One node owns the learner, optionally through persistent CUDA DDP ranks. Uneven
shards are weighted by actual sample counts, preserving the global batch.
BatchNorm statistics remain local to microbatches, so changing GPU count or
microbatch layout can still change learning despite identical sample accounting.
The next CPU batch is prepared during the learner step and finished before
collection can mutate replay. Compiled inference may pad cohorts; learning does
not pad batches because doing so would change BatchNorm statistics.

Local workers and remote nodes receive finite assignments carrying game IDs,
deterministic seeds, model publication IDs, and attempt tokens. Missing games are
retried after failure; acceptance deduplicates independently of replay eviction.
Remote messages use bounded JSON headers and numeric payloads. Authentication
does not encrypt traffic; use a trusted network or SSH tunnel.

## Snapshots and observability

Atomic full checkpoints are saved at fully accounted work boundaries. They
contain model, optimizer, replay, prepared batch, publication, RNG states, and
scheduling counters. Portable model snapshots are immutable evaluation inputs.
`storage.load_checkpoint` also resolves serialized record names from before the
package rename without importing an archived runtime. Full checkpoints from
earlier algorithm designs require their archived implementation.

`metrics.jsonl` is authoritative. TensorBoard is a buffered mirror owned by the
coordinator. A single optional evaluator process runs named jobs sequentially,
with separate event directories. Offline and live evaluation share functions for
episode returns, fixed opponents, and self-lag. Results record hashes, protocol,
precision, paired/episode uncertainty, and the evaluated checkpoint's counters.
Completed job identities prevent duplicates after restart. Evaluation failures
are reported without discarding training or the final model.

Rollback archives superseded snapshots, evaluation results, and logs, then
rebuilds the active TensorBoard history from retained JSON records. Training
elapsed time and final evaluation waiting time are reported separately.

## Transfer initialization

Initialization copies selected representation/dynamics/prediction components
into a fresh optimizer and replay. Names, shapes, encodings, and value perspective
must be compatible. Board-size changes preserve shape-compatible tensors
and reset tensors with incompatible spatial dimensions. Adding komi context zero-extends the input
convolution. An import report lists copied, reset, and expanded tensors.
Selecting only representation and dynamics resets the entire prediction
component, including its shape-compatible tensors; the first board-size pilot
uses this explicit option.

Curriculum orchestration, new architectures, freezing strategies, optimizer/loss
expansion, reanalysis, automatic opponent promotion, and Elo fitting remain
deferred. Initialization is a mechanism for later experiments, not a claim that
any transfer strategy improves learning.
