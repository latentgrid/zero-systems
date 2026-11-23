# Zero Systems

Zero Systems is a performance-optimized PyTorch implementation of MuZero for
training agents through self-play. You can train on a single GPU, share learning
across several GPUs, or add machines to generate more self-play games. We
evaluate both how long training takes and how well the trained agents play.

The training loop and models are written in Python and PyTorch. Numba compiles
CPU tree search and the calculation of value targets from replay. On GPUs, Zero Systems
batches model calls across games and keeps the latent states used by search in
device memory, so expanding a node does not require a round trip to the CPU.

Go is supported across board sizes. To use your own environment, the simplest
route is to register it with Gymnasium and select it in the experiment
configuration. The existing adapter handles discrete actions and fixed-size
numeric observations, so the same search and learner work with your task.
Zero Systems also supports a subset of OpenSpiel games.
[Custom environment example](examples/custom_env/README.md)

## Get started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
run a small Go experiment on your CPU:

```bash
git clone https://github.com/latentgrid/zero-systems.git
cd zero-systems
uv run --locked --extra go zero-systems train configs/experiments/smoke.toml configs/hardware/cpu.toml runs/smoke
```

On the first run, uv creates the Python environment and installs the dependencies.
The example plays on a 3×3 board and performs four learner updates. Look in
`runs/smoke/models/` for model snapshots and `runs/smoke/metrics.jsonl` for
training progress.

## Train on your hardware

Each run takes an [experiment configuration](configs/experiments/), a
[hardware configuration](configs/hardware/) and an output directory. The experiment
defines the game, search and learning settings. The hardware configuration
assigns work to devices, so you can move an experiment to another machine
without changing its training settings.

For example, to train on 9×9 Go on one BF16-capable NVIDIA GPU:

```bash
uv run --locked --extra go zero-systems train configs/experiments/go9.toml configs/hardware/single-gpu-bf16.toml runs/go9
```

## Training speed and playing strength

On the same six-GPU machine, Zero Systems completed the 9×9 Go training budget
**4.36× faster than MiniZero** (3.22 vs. 14.03 hours on average). When the final
models played each other, Zero Systems earned **74.45% of the available points**
across five matched training seeds, counting a draw as half a point.
[Benchmark setup and results](docs/benchmarks.md#six-gpu-go-training)

## Documentation

- [Usage guide](docs/usage.md) — configure environments, evaluate models, resume training, run remote actors and transfer weights between boards.
- [Architecture](docs/architecture.md) — follow a game through search, replay and learning.
- [Benchmarks and validation](docs/benchmarks.md) — inspect correctness checks, benchmark conditions and known limitations.

[Citation](CITATION.cff) · [MIT license](LICENSE)
