import os

import pytest

from zero_systems.cli import main


@pytest.mark.parametrize(
    ("command", "role", "alias"),
    [
        ("evaluate", "candidate", "same"),
        ("evaluate", "opponent", "hardlink"),
        ("evaluate-episodes", "candidate", "symlink"),
    ],
)
def test_evaluation_rejects_output_that_aliases_checkpoint(tmp_path, command, role, alias, capsys):
    candidate = tmp_path / "candidate.pt"
    opponent = tmp_path / "opponent.pt"
    candidate.write_bytes(b"candidate")
    opponent.write_bytes(b"opponent")
    source = candidate if role == "candidate" else opponent
    output = tmp_path / "results.json"
    if alias == "same":
        output = source
    elif alias == "hardlink":
        os.link(source, output)
    else:
        output.symlink_to(source)
    args = [command, str(candidate)]
    if command == "evaluate":
        args.append(str(opponent))
    with pytest.raises(SystemExit) as error:
        main([*args, str(output)])
    assert error.value.code == 2
    assert "output" in capsys.readouterr().err
    assert candidate.read_bytes() == b"candidate" and opponent.read_bytes() == b"opponent"


@pytest.mark.parametrize(
    ("flags", "conf", "message"),
    [
        (["--resume", "old.pt", "--initialize", "source.pt"], "", "--initialize"),
        (["--components", "representation"], "", "--components"),
        (
            ["--resume", "old.pt", "--components", "representation"],
            'initialize = "base.pt"',
            "--components",
        ),
    ],
)
def test_train_rejects_ineffective_flags_before_startup(tmp_path, flags, conf, message, capsys):
    experiment = tmp_path / "experiment.toml"
    experiment.write_text(conf)
    with pytest.raises(SystemExit) as error:
        main(["train", str(experiment), "unused.toml", str(tmp_path / "run"), *flags])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_conversion_preserves_source_with_partial_suffix(tmp_path):
    import torch

    from zero_systems.config import Experiment
    from zero_systems.env import task_spec
    from zero_systems.model import build_model
    from zero_systems.reference import convert_checkpoint
    from zero_systems.storage import load_checkpoint

    experiment = Experiment(board_size=3, channels=8, blocks=1, value_hidden=8)
    source = tmp_path / "converted.pt.partial"
    output = tmp_path / "converted.pt"
    torch.save(build_model(experiment, task_spec(experiment)).state_dict(), source)
    original = source.read_bytes()
    convert_checkpoint(source, experiment, output, updates=3)
    assert source.read_bytes() == original
    assert load_checkpoint(output)["progress"]["updates"] == 3
    with pytest.raises(ValueError, match="new output path"):
        convert_checkpoint(source, experiment, source, updates=3)
    assert source.read_bytes() == original


def test_failed_checkpoint_write_is_clean_and_keeps_previous_snapshot(tmp_path):
    from zero_systems.storage import atomic_save, load_checkpoint

    path = tmp_path / "checkpoint.pt"
    atomic_save(path, {"updates": 1})
    atomic_save(path, {"updates": 2})
    with pytest.raises(TypeError):
        atomic_save(path, (x for x in ()))
    assert load_checkpoint(path) == {"updates": 2}
    assert list(tmp_path.iterdir()) == [path]


def test_transfer_components_accept_initializer_from_experiment_file(tmp_path):
    import json
    from pathlib import Path

    import torch

    from zero_systems.config import Experiment, load
    from zero_systems.env import task_spec
    from zero_systems.model import build_model

    configs = Path(__file__).resolve().parents[1] / "configs"
    smoke = configs / "experiments/smoke.toml"
    experiment = tmp_path / "experiment.toml"
    experiment.write_text(smoke.read_text() + '\ninitialize = "base.pt"\n')
    config = load(experiment, Experiment)
    torch.save(build_model(config, task_spec(config)).state_dict(), tmp_path / "base.pt")

    run = tmp_path / "run"
    main(
        [
            "train",
            str(experiment),
            str(configs / "hardware/cpu.toml"),
            str(run),
            "--components",
            "representation",
            "--updates",
            "1",
        ]
    )
    events = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    init = next(event for event in events if event["event"] == "initialize")
    assert init["components"] == ["representation"]
