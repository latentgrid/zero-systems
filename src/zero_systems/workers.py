"""The same finite-job protocol over local pipes and authenticated remote links."""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import os
import queue
import socket
import threading
import time
import traceback
from multiprocessing.connection import Client, Listener

from .config import Experiment, System, from_dict, identity
from .distributed import setup_process
from .env import task_spec
from .model import build_model
from .play import play_games
from .search import Search
from .storage import decode_weights, encode_game, receive, send, source_identity


def worker_loop(connection, experiment: Experiment, system: System, device: str):
    setup_process(system, device, experiment.seed)
    model = build_model(experiment, task_spec(experiment)).to(device)
    search = Search(
        model, experiment, device, system.actor_precision, system.compile, system.gpu_search
    )
    loaded_step = -1
    send_lock = threading.Lock()
    failed = threading.Event()

    def output(message, payload=b""):
        with send_lock:
            send(connection, message, payload)

    output({"kind": "ready", "capacity": system.actor_batch})
    while True:
        try:
            command, payload = receive(connection)
        except EOFError:
            return
        if command["kind"] == "stop":
            return
        if command["kind"] != "work":
            raise ValueError("worker expected a work assignment")
        token = command["token"]
        heartbeat_stop = threading.Event()

        def heartbeat(heartbeat_stop=heartbeat_stop, token=token):
            while not heartbeat_stop.wait(5):
                try:
                    output({"kind": "heartbeat", "token": token})
                except (OSError, EOFError):
                    failed.set()
                    return

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            if payload:
                model.load_state_dict(decode_weights(payload))
                loaded_step = command["step"]
            if loaded_step != command["step"]:
                raise ValueError("worker received the wrong model version")
            ids = command["game_ids"]
            if len(ids) > system.actor_batch or len(set(ids)) != len(ids):
                raise ValueError("invalid assignment size or duplicate IDs")

            def emit(game, token=token):
                output({"kind": "game", "token": token, "game_id": game.game_id}, encode_game(game))

            play_games(search, experiment, ids, loaded_step, emit=emit, cancelled=failed.is_set)
            output({"kind": "done", "token": token})
        except BaseException:
            output({"kind": "error", "token": token, "error": traceback.format_exc()})
            return
        finally:
            heartbeat_stop.set()
            thread.join(timeout=1)


def _local_worker(connection, experiment, system, device):
    try:
        worker_loop(connection, experiment, system, device)
    except BaseException:
        try:
            send(connection, {"kind": "error", "error": traceback.format_exc()})
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class Peer:
    def __init__(self, connection, events, timeout, *, capacity, process=None, device=None):
        self.connection, self.events, self.timeout = connection, events, timeout
        self.capacity, self.process, self.device = capacity, process, device
        self.commands = queue.Queue(maxsize=1)
        self.ready = False
        self.active = None
        self.alive = True
        self.loaded_step = -1
        self.last_progress = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, work, payload):
        self.ready = False
        self.active = work
        self.last_progress = time.monotonic()
        self.commands.put_nowait((work, payload if work["step"] != self.loaded_step else b""))

    def _read(self):
        deadline = time.monotonic() + self.timeout
        while not self.connection.poll(0.2):
            if not self.alive or time.monotonic() > deadline:
                raise TimeoutError("actor connection timed out")
            if self.process is not None and self.process.exitcode is not None:
                raise RuntimeError(f"actor exited: {self.process.exitcode}")
        return receive(self.connection)

    def _run(self):
        try:
            message, payload = self._read()
            if message["kind"] != "ready":
                raise RuntimeError(message.get("error", "actor failed to initialize"))
            self.events.put((self, message, payload))
            while self.alive:
                try:
                    command = self.commands.get(timeout=0.2)
                except queue.Empty:
                    continue
                if command is None:
                    send(self.connection, {"kind": "stop"})
                    return
                work, payload = command
                send(self.connection, work, payload)
                self.loaded_step = work["step"]
                while self.alive:
                    message, payload = self._read()
                    if message["kind"] == "heartbeat":
                        continue
                    self.last_progress = time.monotonic()
                    self.events.put((self, message, payload))
                    if message["kind"] in {"done", "error"}:
                        break
        except BaseException:
            if self.alive:
                self.events.put((self, {"kind": "error", "error": traceback.format_exc()}, b""))

    def close(self):
        try:
            self.commands.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=1)
        self.alive = False
        self.connection.close()
        if self.process is not None:
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)


def parse_address(address):
    host, separator, port = address.rpartition(":")
    if not separator or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError("address must be HOST:PORT")
    return host, int(port)


def secret(system):
    value = os.environ.get(system.auth_environment)
    if not value:
        raise ValueError(f"set {system.auth_environment} for remote actors")
    return value.encode()


class ActorPool:
    def __init__(self, experiment: Experiment, system: System):
        self.experiment, self.system = experiment, system
        self.events = queue.Queue(maxsize=system.queue_batches)
        self.new_peers = queue.Queue()
        self.peers = []
        self.listener = None
        self.closed = False
        try:
            for _ in range(system.actors_per_device):
                for device in system.actor_devices:
                    self._spawn(device)
            if system.listen:
                self.listener = Listener(parse_address(system.listen), authkey=secret(system))
                self.acceptor = threading.Thread(target=self._accept, daemon=True)
                self.acceptor.start()
        except BaseException:
            self.close()
            raise

    def _spawn(self, device):
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=_local_worker, args=(child, self.experiment, self.system, device)
        )
        process.start()
        child.close()
        peer = Peer(
            parent,
            self.events,
            self.system.timeout,
            capacity=self.system.actor_batch,
            process=process,
            device=device,
        )
        self.peers.append(peer)

    def _accept(self):
        while not self.closed:
            connection = None
            try:
                connection = self.listener.accept()
                if not connection.poll(self.system.timeout):
                    connection.close()
                    continue
                hello, _ = receive(connection)
                capacity = hello.get("capacity")
                if (
                    hello.get("protocol") != 4
                    or not isinstance(capacity, int)
                    or not 0 < capacity <= 65536
                ):
                    connection.close()
                    continue
                send(
                    connection,
                    {
                        "kind": "welcome",
                        "experiment": dataclasses.asdict(self.experiment),
                        "identity": identity(self.experiment),
                    },
                )
                peer = Peer(
                    connection, self.events, self.system.timeout, capacity=hello["capacity"]
                )
                self.new_peers.put(peer)
            except (OSError, EOFError, ValueError, mp.AuthenticationError):
                if connection is not None:
                    connection.close()
                if self.closed:
                    return

    def available(self):
        while True:
            try:
                self.peers.append(self.new_peers.get_nowait())
            except queue.Empty:
                break
        return [p for p in self.peers if p.alive and p.ready]

    def wait_local(self):
        """Do not assign an entire first round to whichever process boots first."""
        deadline = time.monotonic() + self.system.timeout
        remote_ready = []
        while any(p.alive and p.device is not None and not p.ready for p in self.peers):
            result = self.poll()
            if result and result[0].device is None and result[1]["kind"] == "ready":
                remote_ready.append(result[1])
            if result and result[1]["kind"] == "error":
                raise RuntimeError(result[1]["error"])
            if time.monotonic() > deadline:
                raise TimeoutError("local actors did not initialize")
        return remote_ready

    def poll(self, timeout=0.1):
        self.available()
        for peer in self.peers:
            if (
                peer.alive
                and not peer.ready
                and time.monotonic() - peer.last_progress > self.system.timeout
            ):
                return (
                    peer,
                    {"kind": "error", "error": "actor made no completed-game progress"},
                    b"",
                )
        try:
            peer, message, payload = self.events.get(timeout=timeout)
        except queue.Empty:
            return None
        if message["kind"] == "ready":
            peer.ready = True
        return peer, message, payload

    def retire(self, peer):
        peer.close()
        if peer.device is not None and not self.closed:
            self._spawn(peer.device)

    def close(self):
        self.closed = True
        if self.listener:
            self.listener.close()
        self.available()
        for peer in self.peers:
            peer.close()


def actor_node(address, system):
    """One network connection and one cached publication per physical actor node."""
    system.validate()
    if not system.actor_devices:
        raise ValueError("actor node needs actor_devices")
    connection = Client(parse_address(address), authkey=secret(system))
    pool = None
    try:
        capacity = system.actor_batch * system.actors_per_device * len(system.actor_devices)
        send(connection, {"protocol": 4, "capacity": capacity})
        welcome, _ = receive(connection)
        experiment = from_dict(welcome["experiment"], Experiment)
        if identity(experiment) != welcome["identity"]:
            raise ValueError("remote experiment identity mismatch")
        pool = ActorPool(experiment, dataclasses.replace(system, listen=None))
        pool.wait_local()
        send(
            connection,
            {
                "kind": "ready",
                "capacity": capacity,
                "hostname": socket.gethostname(),
                "actor_source_sha256": source_identity(),
            },
        )
        publication, published_step = b"", -1
        while True:
            try:
                work, payload = receive(connection)
            except EOFError:
                return
            if work["kind"] == "stop":
                return
            if work["kind"] != "work":
                raise ValueError("actor node expected work")
            if payload:
                publication, published_step = payload, work["step"]
            if published_step != work["step"]:
                raise ValueError("actor node has the wrong publication")
            pending = list(work["game_ids"])
            if len(pending) > capacity or len(set(pending)) != len(pending):
                raise ValueError("invalid actor node assignment")
            delivered, active = set(), {}
            failures = 0
            heartbeat = time.monotonic()
            while pending or active:
                for peer in pool.available():
                    if not pending:
                        break
                    ids, pending = pending[: peer.capacity], pending[peer.capacity :]
                    token = f"{work['token']}:{ids[0]}:{failures}"
                    subwork = {**work, "token": token, "game_ids": ids}
                    active[token] = ids
                    peer.submit(subwork, publication)
                event = pool.poll(0.1)
                if event:
                    peer, message, game_payload = event
                    kind = message["kind"]
                    token = peer.active["token"] if peer.active else None
                    if kind in {"game", "done"} and token not in active:
                        continue
                    if kind == "game":
                        send(connection, {**message, "token": work["token"]}, game_payload)
                        delivered.add(message["game_id"])
                    elif kind == "done":
                        del active[token]
                        peer.active, peer.ready = None, True
                    elif kind == "error":
                        failures += 1
                        pending.extend(i for i in active.pop(token, []) if i not in delivered)
                        pool.retire(peer)
                        if failures > 2:
                            raise RuntimeError(message["error"])
                if time.monotonic() - heartbeat > 5:
                    send(connection, {"kind": "heartbeat", "token": work["token"]})
                    heartbeat = time.monotonic()
            send(connection, {"kind": "done", "token": work["token"]})
    except BaseException:
        try:
            send(connection, {"kind": "error", "error": traceback.format_exc()})
        except (OSError, EOFError):
            pass
        raise
    finally:
        if pool:
            pool.close()
        connection.close()
