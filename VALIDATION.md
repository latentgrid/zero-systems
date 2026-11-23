# Validation

The maintained implementation has focused CPU, CUDA, and remote-execution checks.
Run `uv run pytest -q tests` and `uv run ruff check src/zero_systems tests tools`.
The package consolidation itself changed names and checkpoint loading. Subsequent
learning corrections are described below. Historical source snapshots, full protocols, logs, and plots
are retained locally under ignored `archive/` and `artifacts/` directories.

## Correctness

After package consolidation, **69 tests pass**, including seven CUDA checks
(94.63 seconds). Ruff lint/format checks pass. The wheel contains only the
`zero_systems` package and the `zero-systems` console entry point. All 11
experiment and 13 system presets validate.

The suite covers PUCT/Gumbel agreement, target and player-perspective accounting,
terminal rewards and truncation bootstraps, scalar/structured observations,
kwargs/masks/action offsets, exact game budgets, fractional position credit,
replay eviction, retries, resume, uneven GPU batches, and microbatch accounting.
Monitoring checks cover JSONL/TensorBoard rollback, immutable evaluation,
paired seeds/seats, self-lag, deduplication, final discovery, and evaluator failure.

An additional package-consolidation regression serializes a full checkpoint under
the previous package name, removes that name from the import environment, and
resumes through the maintained loader. Final weights must match uninterrupted
training exactly. Portable snapshots and existing MiniZero/reference weights
remain usable without an archived runtime on the Python path.

The existing 200-update full checkpoint, 1,000-update CartPole portable model,
60,000-update converted reference, and MiniZero anchor were also loaded directly.
At consolidation, ten core algorithm/execution modules were byte-identical to
the pre-rename source; replay targets have since received the correction below.
A fresh physical-node smoke run completed six learner updates with the renamed
package; local and remote source identities match, and all experience is remote.

## Bounded development studies

The monitoring/environment changes were checked against their frozen pre-change
baseline on CartPole and Tic-Tac-Toe: 1,000 updates, global batch 64, development
seeds 303/304/305, and evaluations at 0/500/1,000. All 18 corresponding model
snapshots were bitwise equal, with identical game/position/sample counters and
evaluation results. Each run consumed 64,000 learner roots and 2,016 games.

| Task | Initial mean | Final mean | Protocol |
| --- | ---: | ---: | --- |
| CartPole return | 9.46 | 70.18 | 30 shared evaluation seeds per checkpoint |
| Tic-Tac-Toe score vs random | 51.17% | 56.92% | 200 games in 100 seat-swapped opening pairs |
| Tic-Tac-Toe score vs initialization | 50.00% | 55.75% | Same paired protocol |

Individual seeds are unstable: CartPole seed 304 collapses after improving, and
Tic-Tac-Toe seed 305 regresses against random. The frozen baseline behaves
identically. These are bounded regression checks, not evidence of reliably solved
tasks or general sample-efficiency equivalence. Reserved seeds 401–403 remain
untouched. Detailed per-seed intervals and raw returns are preserved in the local
measurement archive.

### Terminal targets and PUCT scaling

A follow-up found that generic games preserved terminal rewards but masked
all subsequent value/reward supervision. Search could therefore predict
additional return beyond termination. Generic terminal unrolls now learn an
absorbing tail: zero remaining value, zero further reward and uniform policy.
Truncation bootstraps and Go's separate outcome convention are unchanged.
Generic PUCT presets also enable the existing Q normalization option. On the original
collapsed seed-304 checkpoint, this inference-only change raises return from
9.5 to 154.97 without changing weights.

The fixed 1,000-update study was repeated on development seeds 303–305:

| Task | Original final mean | Corrected final mean | Corrected per-seed finals |
| --- | ---: | ---: | --- |
| CartPole return | 70.18 | 212.59 | 195.87 / 248.23 / 193.67 |
| Tic-Tac-Toe score vs random | 56.92% | 62.17% | 62.00% / 64.75% / 59.75% |
| Tic-Tac-Toe score vs initialization | 55.75% | 57.92% | 56.25% / 62.25% / 55.25% |

The normalization-only CartPole control averaged 263.40, with a wider final
range of 79.60–500.00. The terminal correction addresses a missing target; it
does not improve every seed or every checkpoint. Longer CartPole episodes also
increase accepted experience: approximately 115,269 positions per corrected run
versus 45,448 originally, at the same 64,000 sampled roots. These results do not
establish superiority at equal environment samples or reliably solved tasks.

An exhaustive Tic-Tac-Toe diagnostic covers all 4,520 nonterminal states and
integrates score against random play exactly. It confirms the original seed-305
regression was not solely arena sampling noise. The corrected final policies
improve over initialization on all three seeds under this exact diagnostic.
Focused tests cover zero absorbing targets, preserved terminal rewards and
unmodified truncation handling. All fields in 128 fixed Go replay samples are
bitwise equal before and after the correction, including symmetry augmentation.

The companion study preserves protocols, per-seed intervals, raw evaluations,
source snapshots and plotting commands in
`Project-Data-Science/experiments/04-stability-transfer/`.
After this correction, all 62 CPU checks and seven CUDA checks pass; the CUDA
portion takes 45.52 seconds. Generic normalized-PUCT presets also complete short
CLI training checks on CartPole async, Taxi and FrozenLake.

### Initial board-size transfer pilot

Three development seeds compare 600 updates on 5×5 plus 1,200 on 9×9 with
scratch training for 1,200 or 1,800 updates. Representation and dynamics transfer;
the entire prediction component, optimizer and replay start fresh. All nine
training stages finish with exact game/sample budgets and verified initialization.

The early MiniZero anchor saturates and the mature 60k anchor wins every game
at this budget. Direct 200-game comparisons are more informative: transfer
scores 100.0% / 98.5% / 66.5% against scratch at equal target updates, and
96.5% / 74.75% / 38.5% at equal total learner samples including pretraining.
This is a promising adaptation result, with mixed total-cost benefit. Transfer
uses 17.3% fewer accepted positions but 7.0% more elapsed time on average; setup,
cache warming and the stage-dependent resignation schedule limit interpretation.
No transfer curriculum becomes a default on this evidence, and no new claim of
MiniZero sample-efficiency parity is made. The companion report contains all
paired intervals, curves, costs and the declared additional evaluation protocol.

## Systems measurements

Go 9×9 measurements use the fixed-offset 256-channel/3-block network, Gumbel-16,
BF16 learning/inference, global batch 1,024, and microbatch 256. Fixed-input
components use five timed repetitions after three warmups. End-to-end checks
use three warmed 50-update windows in 250-update/2,500-game runs.

| Topology | Monitoring off: training time | TensorBoard | TensorBoard + live evaluation |
| --- | ---: | ---: | ---: |
| One RTX 3090 shared by learner/actors | 210.72 s | 208.11 s | 228.29 s |
| Four learner GPUs + two actor GPUs | 144.92 s | 146.62 s | 148.68 s |

The multi-GPU learner uses two RTX 3090 and two RTX 3080 GPUs; actors use two
RTX 3080 GPUs. No monitoring-disabled regression above 5% was observed versus
the frozen pre-change baseline. TensorBoard shows no measured single-GPU
slowdown and 1.17% elapsed overhead on the multi-GPU workload; short-run timing
variation prevents interpreting this as a precise overhead confidence bound.

Live evaluation deliberately used an accelerated cadence of 100 games every
100 updates on a shared learner GPU. Final evaluation added a separate 67.87 s
single-GPU / 71.24 s multi-GPU wait. The normal Go preset uses 6,000 updates;
resource-time estimates at that cadence are roughly 0.8–2.4% and 1.7–2.0%.
These are estimates, not measured amortized overhead over a complete interval.

A physical remote RTX 3080/2060 worker was restarted during position-paced
training, interrupting 32 assigned games. Retries completed successfully. Full
resume advanced from 200 to 400 updates, reaching 51,200 learner roots and 51,891
unique accepted positions, all remote, with no duplicate credit. Owned containers,
tunnels, and MPS services were cleaned up afterward.

An earlier 60,000-update Go development run at 600,000 games scored 50.56%
against the self-trained MiniZero anchor in common search (paired 95% interval
47.13–54.00%) and 49.0% against the native engine (42.5–55.5%). Runtime was 0.57%
longer than the retained original implementation; that implementation remained
stronger in direct play. This is a single-seed endpoint comparison. No new long
training run was needed for the package consolidation.

## Limits

The live watcher uses Linux file locking. BatchNorm can change learning with GPU
or microbatch layout even when sample counters agree. Evaluation intervals apply
to the specified opponent, search, and opening protocol, not universal strength.
Curriculum orchestration and broader architecture/optimizer experiments remain
deferred. Full original protocols and source fingerprints remain in the local
archive so historical measurements are not silently relabeled as new runs.
