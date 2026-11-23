# Train on a custom Gymnasium environment

[Line Walk](line_walk.py) starts the agent in the middle of five positions.
Each action moves one position left or right. Reaching the right end earns
+1 and reaching the left end earns −1. Gymnasium ends the episode at either
endpoint or truncates it after ten moves.

Run these commands from the repository root:

```bash
export PYTHONPATH="$PWD/examples/custom_env${PYTHONPATH:+:$PYTHONPATH}"
uv run --locked --extra gym zero-systems train examples/custom_env/experiment.toml configs/hardware/cpu.toml runs/line-walk
uv run --locked --extra gym zero-systems evaluate-episodes runs/line-walk/models/step_00000004.pt runs/line-walk/returns.json --episodes 10
```

The [experiment](experiment.toml) selects `line_walk:LineWalk-v0`. Gymnasium
imports `line_walk` in each process before looking up the registered ID.
`PYTHONPATH` makes the example module available to training workers and to
the separate evaluation command. For an installed environment package, use
its importable module name in place of `line_walk`.

The CPU run collects eight episodes and performs four updates. To adapt it,
implement Gymnasium's `reset` and `step` methods, register your environment,
then change the module and ID in the experiment file. The adapter accepts
discrete actions and fixed-size numeric observations. See the
[usage guide](../../docs/usage.md) for supported spaces and action masks.
