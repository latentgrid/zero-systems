# Benchmarks

## Six-GPU Go training

The main thesis experiment compared Zero Systems and MiniZero on 9×9 Go.
Each run played 600,000 games and completed 60,000 learner updates with a global
batch of 1,024. Each implementation ran five training seeds on the same machine
with four RTX 3080s and two RTX 3090s.

| Measurement | Result |
| --- | ---: |
| Mean complete training time, Zero Systems | 3.22 hours |
| Mean complete training time, MiniZero | 14.03 hours |
| End-to-end speedup (ratio of mean runtimes) | 4.36× |
| Final Zero Systems score in direct play against MiniZero | 74.45% |

**Training conditions.** Both implementations used Gumbel search with 16
simulations per move and all six GPUs for self-play. Zero Systems collected
games concurrently with four learner GPUs using BF16 mixed precision.
MiniZero alternated self-play with learning on one RTX 3090 using FP32. The
speedup measures the combined effect of these configurations. Training time
includes initialization, compilation where applicable, checkpointing, recovery
work and shutdown. Offline evaluation and idle gaps between attempts are excluded.

**Playing strength.** Each final Zero Systems model played 800 games against
the final MiniZero model from the same training seed. Both sides used the same
FP32 evaluator with 400 Gumbel simulations per move. Each opening was played
with both seat assignments. A win earned one point and a draw half a point.
Individual seed scores ranged from 48.56% to 100%. The reported mean applies
to these five trained model pairs and this evaluation protocol.

## Running checks locally

The [test suite](../tests/) covers search, replay targets, scheduling, restart
behavior, distributed execution and model transfer. From the repository root:

```bash
uv run --locked --extra dev pytest -m "not cuda" tests
uv run --locked --extra dev ruff check src tests tools
```

To time individual search and learner operations on your own hardware, run the
fixed-input benchmark. Its warmup and timed repetitions measure a different
workload from complete training:

```bash
uv run --locked --extra go zero-systems benchmark configs/experiments/go9.toml configs/hardware/single-gpu-bf16.toml results/benchmark.json --warmup 3 --repeats 5
```
