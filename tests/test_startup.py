import socket
from dataclasses import replace

import pytest
from test_core import small

from zero_systems.config import System
from zero_systems.run import train


def test_failed_checkpoint_load_leaves_destination_reusable(tmp_path):
    config, system = small(updates=1), System(actor_batch=4)
    directory = tmp_path / "target"
    with pytest.raises(FileNotFoundError):
        train(config, system, directory, resume=tmp_path / "missing.pt")
    assert not (directory / "metrics.jsonl").exists()
    assert train(config, system, directory)["updates"] == 1


def test_failed_actor_setup_preserves_active_run_on_rollback(tmp_path, monkeypatch):
    config, system = small(updates=1), System(actor_batch=4)
    directory = tmp_path / "run"
    train(config, system, directory)
    initial = directory / "checkpoints/step_00000000.pt"
    latest = directory / "models/step_00000001.pt"
    journal = directory / "metrics.jsonl"
    original_history = journal.read_bytes()

    monkeypatch.setenv(system.auth_environment, "test-key")
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        busy_system = replace(system, listen=f"127.0.0.1:{occupied.getsockname()[1]}")
        with pytest.raises(OSError):
            train(config, busy_system, directory, resume=initial)

    assert latest.is_file()
    assert journal.read_bytes() == original_history
    assert not (directory / "history").exists()


def test_resume_rejects_a_new_initialization_checkpoint(tmp_path):
    config, system = small(updates=1), System(actor_batch=4)
    directory = tmp_path / "run"
    train(config, system, directory)
    checkpoint = directory / "checkpoints/step_00000001.pt"
    with pytest.raises(ValueError, match="initialization"):
        train(replace(config, initialize="another-model.pt"), system, directory, resume=checkpoint)
    assert not (directory / "history").exists()
