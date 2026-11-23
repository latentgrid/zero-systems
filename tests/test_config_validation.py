import pytest

from zero_systems.config import Experiment, System, load


@pytest.mark.parametrize(
    ("schema", "source", "field"),
    [
        (Experiment, "batch_size = 16.5", "batch_size"),
        (Experiment, "simulations = true", "simulations"),
        (Experiment, "komi = [true]", "komi"),
        (System, 'compile = "false"', "compile"),
        (System, 'actor_devices = "cpu"', "actor_devices"),
        (System, "learner_devices = [true]", "learner_devices"),
        (System, '[[evaluations]]\nname = "anchor"\ninterval_updates = 1.5', "interval_updates"),
        (System, '[[evaluations]]\nname = "anchor"\ngamez = 10', "gamez"),
        (System, 'evaluations = ["anchor"]', "evaluations"),
        (System, "[[evaluations]]\ngames = 10", "name"),
    ],
)
def test_toml_reports_invalid_setting(tmp_path, schema, source, field):
    path = tmp_path / "preset.toml"
    path.write_text(source)
    with pytest.raises(ValueError, match=field):
        load(path, schema)


def test_integer_literals_are_valid_for_float_settings(tmp_path):
    path = tmp_path / "preset.toml"
    path.write_text("komi = [7]\ndiscount = 1")
    experiment = load(path, Experiment)
    assert experiment.komi == (7.0,)
    assert experiment.discount == 1.0


@pytest.mark.parametrize(
    ("source", "field"),
    [
        ('learner_devices = ["cuda:bad"]', "learner_devices"),
        ('actor_devices = ["gp0"]', "actor_devices"),
        ('[[evaluations]]\nname = "eval"\ndevice = "gpu0"', "device"),
        ('learner_devices = ["cuda", "cuda:0"]', "learner_devices"),
        ('learner_devices = ["cuda:0", "cuda:256"]', "learner_devices"),
        ('precision = "bfloat16"', "precision"),
        ('[[evaluations]]\nname = "eval"\nprecision = "bfloat16"', "precision"),
    ],
)
def test_bad_device_configuration_fails_on_load(tmp_path, source, field):
    path = tmp_path / "preset.toml"
    path.write_text(source)
    with pytest.raises(ValueError, match=field):
        load(path, System)


@pytest.mark.parametrize("entrypoint", ["train", "benchmark"])
def test_incompatible_batch_layout_fails_before_startup(tmp_path, entrypoint):
    from zero_systems.benchmark import benchmark
    from zero_systems.run import train

    experiment = Experiment(batch_size=7, environment="Unregistered-v0", augmentation=False)
    system = System(learner_devices=tuple(f"cuda:{i}" for i in range(6)), microbatch=1)
    with pytest.raises(ValueError, match="batch_size.*learner.*microbatch"):
        if entrypoint == "train":
            train(experiment, system, tmp_path / "run")
        else:
            benchmark(experiment, system)
    assert not (tmp_path / "run").exists()
