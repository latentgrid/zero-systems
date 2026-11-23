# Usage guide

Run the commands below from the repository root. The
[CPU quickstart](../README.md#get-started) installs the Go dependencies and
runs a four-update training job. Use a new output directory for each experiment.

## Choose a training configuration

`zero-systems train EXPERIMENT HARDWARE RUN` takes two TOML files.
The [experiment presets](../configs/experiments/) set the environment, network,
search and learning budget. The [hardware presets](../configs/hardware/) set
device placement, batch limits, checkpoint cadence and monitoring.

| Goal | Experiment | Hardware or prerequisite |
| --- | --- | --- |
| Check the installation | `smoke.toml` | `cpu.toml` |
| Train on 9×9 Go | `go9.toml` (Gumbel search) or `go9-puct.toml` | `single-gpu.toml` or `single-gpu-bf16.toml` on a BF16-capable GPU |
| Run a single-player Gymnasium task | `cartpole.toml`, `cartpole-async.toml`, `frozenlake.toml`, `taxi.toml` | `cpu.toml`, or `cpu-monitored.toml` for TensorBoard and episode evaluation |
| Train on an OpenSpiel board game | `tictactoe.toml`, `connect-four.toml` | `cpu.toml` |
| Work with MiniZero reference weights | `go9-reference.toml` | Original head for converting compatible external weights |
| Explore the 60k-update development recipe | `go9-validation60k.toml` | Inspect device indices before using a multi-GPU preset. See [benchmark conditions](benchmarks.md) for the actual thesis runs. |

For example, to train on CartPole and view its TensorBoard metrics:

```bash
uv run --locked --extra gym zero-systems train configs/experiments/cartpole.toml configs/hardware/cpu-monitored.toml runs/cartpole
uv run --locked --extra gym tensorboard --logdir runs/cartpole/tensorboard
```

Use `--extra go` for Go and OpenSpiel. The `single-gpu-monitored.toml` preset evaluates the
current Go model against a snapshot from 6,000 updates earlier. To use a fixed
reference model, set the evaluation mode to `"fixed"` and supply its path as
`opponent`.

## Environment contract

To add your own game or control task, follow the [custom Gymnasium example](../examples/custom_env/README.md).
Gymnasium tasks use their registered environment ID, for example
`environment = "CartPole-v1"`, with `network = "mlp"` and
`augmentation = false`. The adapter accepts discrete actions and fixed-size
numeric observations, including supported Dict and Tuple spaces. It flattens
observations and translates nonzero-based action IDs for search. Pass optional
constructor arguments in an `[environment_kwargs]` table. Action masks can come
from `info["action_mask"]` or a named Dict observation field configured with
`action_mask_source = "observation"` and `action_mask_key`.

For OpenSpiel, use `environment = "spiel:GAME_NAME"` for deterministic,
perfect-information games with alternating turns and one player or two zero-sum
players. The dedicated `environment = "go"` adapter also handles observation
history, symmetry augmentation, komi and resignation.

## Schedule self-play and learning

Use `schedule = "phased"` to collect `games_per_round` games and perform
`updates_per_round` learner updates. With `schedule = "async"`, actors collect
games while the learner works. Set `games_per_update` to pace learning by
completed games, or `samples_per_position` to pace it by collected decision
positions. Each update samples a global batch from replay. One sampled
position and its unroll count as one sample across all learner devices.

For position-paced runs, `warmup_positions` sets when learning can begin and
`position_lead` limits how far collection can get ahead. The optional
`--max-positions` budget stops new assignments. Games already in progress can
bring the final count above the requested budget.

## Evaluate a checkpoint

Evaluate a portable snapshot from `RUN/models/`. For example, after the CPU
quickstart, play the final Go model against random moves:

```bash
uv run --locked --extra go zero-systems evaluate runs/smoke/models/step_00000004.pt random results/smoke.json --games 20 --opening-moves 0
```

Replace `random` with a compatible opponent snapshot for a direct match.
Competitive evaluation uses paired games with reversed player seats.
For single-player tasks, compare episode returns. For example:

```bash
uv run --locked --extra gym zero-systems evaluate-episodes runs/cartpole/models/step_00001000.pt results/cartpole.json --episodes 30 --seed 17301
```

For evaluation during training, add `[[evaluations]]` entries to the hardware
TOML. Choose `mode = "episodes"` for single-player returns, `"fixed"` for a
specified opponent or `"self_lag"` for an older snapshot from the run.
Progress and errors are recorded in `evaluations/status.json`.
Training metrics are always available in `metrics.jsonl`.

## Resume training

Use a full checkpoint from `RUN/checkpoints/` to continue with the saved
optimizer, replay and progress counters. The two most recent full checkpoints
are retained. To extend the quickstart to eight total updates:

```bash
uv run --locked --extra go zero-systems train configs/experiments/smoke.toml configs/hardware/cpu.toml runs/smoke --resume runs/smoke/checkpoints/step_00000004.pt --updates 8
```

Resuming an earlier checkpoint archives later model snapshots, results and
logs under `RUN/history/`. Load full checkpoints only from a trusted source.

## Add remote self-play workers

Install the same source on both machines and set `ZERO_SYSTEMS_AUTHKEY` to
the same private value in both shells. Use a trusted network or an SSH tunnel
because the transport authenticates connections without encrypting them.
The supplied learner preset listens on port 47631 and relies on remote actors
for self-play.

On the learner host:

```bash
uv run --locked --extra go zero-systems train configs/experiments/go9.toml configs/hardware/remote-learner.toml runs/remote
```

On an actor host, replace `LEARNER_HOST` with its reachable hostname or address:

```bash
uv run --locked --extra go zero-systems actors LEARNER_HOST:47631 configs/hardware/remote-actors.toml
```

To run the actor in Docker on a Linux GPU host:

```bash
docker build -t zero-systems:local .
docker run --rm --network host --gpus all --shm-size=2g -e ZERO_SYSTEMS_AUTHKEY zero-systems:local actors LEARNER_HOST:47631 configs/hardware/remote-actors.toml
```

The optional [validation launcher](../tools/run_multinode.py) automates SSH, Docker
and MPS for the development hardware. Its defaults are specific to that setup.

## Transfer a model

Start a new training directory and pass `--initialize SOURCE.pt` to `train`.
By default, representation, dynamics and prediction are selected. To train
a new prediction head while retaining the other components, pass
`--components representation dynamics`. The new stage gets a fresh optimizer
and replay buffer, and all parameters can continue learning.

For Go transfer across board sizes, set `prediction_head = "spatial"` in both
experiments. All selected weights and normalization buffers must be compatible.
The dense head can instead reset tensors whose shapes depend on the old board
size. The saved initialization record lists which tensors were copied or reset.
See the [model overview](architecture.md#board-size-transfer).
