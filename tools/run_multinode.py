"""Run this repository's actors on the available SSH/Docker worker node.

Only the source and actor config are copied. The existing image supplies
Python/CUDA dependencies; its installed entry point is never invoked.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from zero_systems.config import System, load
from zero_systems.storage import source_identity, versions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--updates", type=int, default=60000)
    parser.add_argument("--resume")
    parser.add_argument(
        "--restart-remote-after-games",
        type=int,
        default=0,
        help="diagnostic: restart this run's remote container once after N remote games",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Profile the coordinator with cProfile"
    )
    parser.add_argument("--experiment", default="configs/experiments/go9-validation60k.toml")
    parser.add_argument("--system", default="configs/hardware/multinode-validation.toml")
    parser.add_argument("--remote-system", default="configs/hardware/remote-validation.toml")
    parser.add_argument("--remote", default="root@192.168.1.10")
    parser.add_argument("--remote-root", default="/mnt/user/appdata/zero-systems-validation")
    parser.add_argument("--image", default="zero-systems:multinode-next")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    directory = Path(args.directory).resolve()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    logs = directory / "launches" / stamp
    logs.mkdir(parents=True, exist_ok=False)
    system = load(args.system, System)
    host, port_text = system.listen.rsplit(":", 1)
    port = int(port_text)
    remote_root = args.remote_root.rstrip("/")
    container = "zero-systems-" + stamp
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.remote]

    def remote(*command):
        return subprocess.run(
            ssh + [shlex.join(command)], check=True, capture_output=True, text=True
        )

    remote("mkdir", "-p", remote_root)
    archive = logs / "actor-source.tgz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted((root / "src/zero_systems").glob("*.py")):
            bundle.add(path, arcname=str(path.relative_to(root)))
        bundle.add(args.remote_system, arcname="actor-system.toml")
    subprocess.run(
        ["scp", "-q", str(archive), f"{args.remote}:{remote_root}/source.tgz"], check=True
    )
    remote("tar", "xzf", remote_root + "/source.tgz", "-C", remote_root)
    docker = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--gpus",
        "all",
        "--shm-size=2g",
        "--mount",
        f"type=bind,source={remote_root},target=/opt/zero-worker,readonly",
        "-e",
        "PYTHONPATH=/opt/zero-worker/src",
        "--entrypoint",
        "python",
    ]
    probe = (
        "import json,torch,socket; from zero_systems.storage import versions; "
        "print(json.dumps(dict(versions=versions(),host=socket.gethostname(),"
        "gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])))"
    )
    remote_metadata = json.loads(remote(*(docker + [args.image, "-c", probe])).stdout)
    if remote_metadata["versions"]["source_sha256"] != source_identity():
        raise RuntimeError("remote source differs from local source")
    metadata = dict(local=versions(), remote=remote_metadata, arguments=vars(args))
    (logs / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    auth = logs / "auth.env"
    token = secrets.token_hex(32)
    descriptor = os.open(auth, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write("ZERO_SYSTEMS_AUTHKEY=" + token + "\n")
    remote_auth = remote_root + "/auth-" + stamp + ".env"
    subprocess.run(["scp", "-q", str(auth), f"{args.remote}:{remote_auth}"], check=True)
    environment = dict(
        os.environ,
        ZERO_SYSTEMS_AUTHKEY=token,
        CUDA_MPS_PIPE_DIRECTORY="/tmp/zero-systems-validation-mps",
        CUDA_MPS_LOG_DIRECTORY=str(logs / "mps"),
    )
    handles = []
    learner = worker = tunnel = None
    mps = False

    def logfile(name):
        handle = (logs / name).open("w")
        handles.append(handle)
        return handle

    try:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env=environment, check=True)
        mps = True
        tunnel = subprocess.Popen(
            ssh[:-1]
            + [
                "-o",
                "ExitOnForwardFailure=yes",
                "-N",
                "-R",
                f"127.0.0.1:{port}:127.0.0.1:{port}",
                args.remote,
            ],
            stdout=logfile("tunnel.log"),
            stderr=subprocess.STDOUT,
        )
        command = [
            sys.executable,
            "-m",
            "zero_systems",
            "train",
            args.experiment,
            args.system,
            str(directory),
            "--updates",
            str(args.updates),
        ]
        if args.resume:
            command += ["--resume", args.resume]
        if args.profile:
            command[1:1] = ["-m", "cProfile", "-o", str(logs / "learner.prof")]
        learner = subprocess.Popen(
            command, env=environment, stdout=logfile("learner.log"), stderr=subprocess.STDOUT
        )
        deadline = time.monotonic() + system.timeout
        while True:
            if learner.poll() is not None or tunnel.poll() is not None:
                raise RuntimeError("learner or SSH tunnel exited during startup")
            try:
                with socket.create_connection((host, port), timeout=1):
                    break
            except OSError as error:
                if time.monotonic() > deadline:
                    raise TimeoutError("learner listener did not start") from error
                time.sleep(1)
        worker_command = docker + [
            "--name",
            container,
            "--env-file",
            remote_auth,
            args.image,
            "-m",
            "zero_systems",
            "actors",
            f"127.0.0.1:{port}",
            "/opt/zero-worker/actor-system.toml",
        ]
        worker = subprocess.Popen(
            ssh + [shlex.join(worker_command)],
            stdout=logfile("remote.log"),
            stderr=subprocess.STDOUT,
        )
        print(
            json.dumps(dict(event="started", learner_pid=learner.pid, logs=str(logs))), flush=True
        )
        worker_finished = None
        restarted = False
        while learner.poll() is None:
            if args.restart_remote_after_games and not restarted:
                metrics = directory / "metrics.jsonl"
                records = metrics.read_text().splitlines() if metrics.exists() else []
                progress = {}
                for line in reversed(records):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # The learner may still be writing the last record.
                    if "remote_games" in record:
                        progress = record
                        break
                if progress.get("remote_games", 0) >= args.restart_remote_after_games:
                    remote("docker", "rm", "-f", container)
                    worker.wait(timeout=30)
                    worker = subprocess.Popen(
                        ssh + [shlex.join(worker_command)],
                        stdout=logfile("remote-restarted.log"),
                        stderr=subprocess.STDOUT,
                    )
                    restarted = True
                    print(json.dumps(dict(event="remote_restart", progress=progress)), flush=True)
            if tunnel.poll() is not None or worker.poll() not in (None, 0):
                raise RuntimeError("remote worker or tunnel failed; stop at a recoverable boundary")
            if worker.poll() == 0:
                worker_finished = worker_finished or time.monotonic()
                if time.monotonic() - worker_finished > system.timeout:
                    raise RuntimeError("remote actors stopped before the learner finished")
            time.sleep(5)
        if learner.returncode:
            raise RuntimeError(f"learner exited with status {learner.returncode}; see {logs}")
        final = json.loads((directory / "metrics.jsonl").read_text().splitlines()[-1])
        print(
            json.dumps(
                dict(
                    event="completed",
                    updates=final["updates"],
                    reason=final["reason"],
                    logs=str(logs),
                )
            ),
            flush=True,
        )
    finally:
        if learner is not None and learner.poll() is None:
            learner.send_signal(signal.SIGINT)
            try:
                learner.wait(timeout=system.timeout + 30)
            except subprocess.TimeoutExpired:
                learner.kill()
                learner.wait()
        subprocess.run(
            ssh + [shlex.join(["docker", "rm", "-f", container])],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if worker is not None:
            try:
                worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker.terminate()
                worker.wait(timeout=15)
        if tunnel is not None:
            tunnel.terminate()
            tunnel.wait(timeout=15)
        if mps:
            subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env=environment)
        subprocess.run(ssh + [shlex.join(["rm", "-f", remote_auth])], check=False)
        auth.unlink(missing_ok=True)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
