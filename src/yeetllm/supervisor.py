from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import httpx

from yeetllm.adapters import resolve_model_adapters
from yeetllm.cluster import (
    ClusterCoordinator,
    ClusterInfo,
    detect_cluster,
    poll_coordinator,
    validate_cluster_ports,
)
from yeetllm.commands import PERSISTENT_MODEL_DOWNLOAD_DIR, build_engine_launch
from yeetllm.config import (
    YeetConfig,
    discover_gpu_count,
    installed_quantization_methods,
    local_adapter_rank,
    normalized_config_hash,
    validate_runtime,
)
from yeetllm.processes import ManagedProcess
from yeetllm.registry import Registry, build_registry
from yeetllm.ssh import SSHLaunch, prepare_ssh
from yeetllm.state import DEFAULT_STATE_PATH, StateStore

PERSISTENT_DIRECTORIES = (
    Path("/workspace/yeetllm/cache/huggingface"),
    Path("/workspace/yeetllm/cache/vllm"),
    Path("/workspace/yeetllm/models"),
    Path("/workspace/yeetllm/quantized"),
)
STARTUP_PROGRESS_INTERVAL_SECONDS = 30.0


class Supervisor:
    def __init__(
        self,
        config: YeetConfig,
        *,
        state_path: Path = DEFAULT_STATE_PATH,
        cluster: ClusterInfo | None = None,
        gpu_count: int | None = None,
    ) -> None:
        self.config = config
        self.cluster = cluster or detect_cluster(config.cluster)
        self.gpu_count = discover_gpu_count() if gpu_count is None else gpu_count
        self.registry: Registry = build_registry(config)
        self.state = StateStore(state_path)
        self.config_hash = normalized_config_hash(config)
        self.stop_event = asyncio.Event()
        self.processes: dict[str, ManagedProcess] = {}
        self.coordinator: ClusterCoordinator | None = None
        self.adapter_paths: dict[str, dict[str, Path]] = {}
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._had_failure = False

    async def run(self) -> int:
        self._validate()
        self._prepare_directories()
        self._initialize_state()
        self._install_signal_handlers()
        self._print_summary()
        self._heartbeat_task = asyncio.create_task(self._heartbeat())
        try:
            if self.cluster.primary:
                return await self._run_primary()
            return await self._run_worker()
        finally:
            await self._shutdown()

    def _validate(self) -> None:
        validate_cluster_ports(
            self.cluster,
            len(self.config.models),
            reserved_ports=self._reserved_listener_ports(),
        )
        if self.cluster.enabled and self.gpu_count != self.cluster.trainers_per_node:
            raise ValueError(
                f"NUM_TRAINERS={self.cluster.trainers_per_node}, but this node exposes "
                f"{self.gpu_count} GPUs"
            )
        warnings = validate_runtime(
            self.config,
            gpu_count=self.gpu_count,
            node_count=self.cluster.node_count,
            quantization_methods=installed_quantization_methods(),
            rank_resolver=local_adapter_rank,
        )
        for warning in warnings:
            print(f"[yeetllm] WARNING: {warning}", flush=True)

    def _prepare_directories(self) -> None:
        for directory in PERSISTENT_DIRECTORIES:
            directory.mkdir(parents=True, exist_ok=True)
        runtime = self.state.path.parent
        runtime.mkdir(parents=True, exist_ok=True)
        if os.geteuid() == 0:
            import pwd

            try:
                account = pwd.getpwnam("vllm")
            except KeyError:
                return
            for directory in PERSISTENT_DIRECTORIES:
                os.chown(directory, account.pw_uid, account.pw_gid)

    def _initialize_state(self) -> None:
        role = "primary" if self.cluster.primary else "worker"
        self.state.initialize(
            {
                "version": 1,
                "node": {
                    "role": role,
                    "rank": self.cluster.node_rank,
                    "count": self.cluster.node_count,
                    "world_size": self.cluster.world_size or self.gpu_count,
                    "address": self.cluster.node_addr,
                },
                "supervisor": {
                    "pid": os.getpid(),
                    "phase": "starting",
                    "started_at": time.time(),
                },
                "registry": self.registry.as_state(),
                "engines": {
                    identifier: {"state": "pending", "pid": None, "error": None}
                    for identifier in self.registry.engines
                },
                "router": {
                    "enabled": self.cluster.primary,
                    "state": "pending",
                    "pid": None,
                    "host": self.config.server.host,
                    "port": self.config.server.port,
                },
                "ssh": {
                    "enabled": False,
                    "state": "disabled",
                    "pid": None,
                    "port": self.config.ssh.port,
                    "key_source": None,
                },
                "cluster": {
                    "enabled": self.cluster.enabled,
                    "interface": self.cluster.interface or None,
                    "primary_addr": self.cluster.primary_addr if self.cluster.enabled else None,
                    "control_port": self.cluster.control_port if self.cluster.enabled else None,
                },
            }
        )

    async def _run_primary(self) -> int:
        if self.cluster.enabled:
            self.coordinator = ClusterCoordinator(
                self.cluster, self.config_hash, set(self.registry.engines)
            )
            await self.coordinator.start()
            print(
                f"[cluster] coordinator listening on {self.cluster.primary_addr}:"
                f"{self.cluster.control_port}",
                flush=True,
            )

        await self._start_ssh()
        await self._start_router()
        self.state.data["supervisor"]["phase"] = "loading"
        self.state.touch()

        failed = False
        model_indices = list(range(len(self.config.models)))
        for batch in chunked(model_indices, self.config.startup.parallelism):
            if self.stop_event.is_set():
                break
            results = await asyncio.gather(
                *(self._prepare_model(index) for index in batch), return_exceptions=True
            )
            ready_indices: list[int] = []
            for index, result in zip(batch, results, strict=True):
                if isinstance(result, BaseException):
                    self._mark_engine_failed(self.config.models[index].id, result)
                    failed = True
                else:
                    ready_indices.append(index)

            if self.coordinator is not None and ready_indices:
                desired = [
                    self.config.models[index].id
                    for index in range(max(ready_indices) + 1)
                    if self.state.data["engines"][self.config.models[index].id]["state"]
                    != "failed"
                ]
                self.coordinator.set_desired(desired)
                try:
                    await self.coordinator.wait_workers(
                        [self.config.models[index].id for index in ready_indices],
                        self.config.cluster.startup_timeout_seconds,
                    )
                except Exception as exc:  # noqa: BLE001 - converts startup failure to state
                    for index in ready_indices:
                        self._mark_engine_failed(self.config.models[index].id, exc)
                    self._sync_coordinator_desired()
                    failed = True
                    continue

            launch_results = await asyncio.gather(
                *(self._launch_primary_engine(index) for index in ready_indices),
                return_exceptions=True,
            )
            for index, result in zip(ready_indices, launch_results, strict=True):
                if isinstance(result, BaseException):
                    self._mark_engine_failed(self.config.models[index].id, result)
                    process = self.processes.get(f"engine:{self.config.models[index].id}")
                    if process is not None:
                        await process.stop()
                    failed = True
            self._sync_coordinator_desired()

        all_ready = all(
            value["state"] == "ready" for value in self.state.data["engines"].values()
        )
        if all_ready and not self.stop_event.is_set():
            self.state.data["supervisor"]["phase"] = "ready"
            self.state.touch()
            print("[yeetllm] all required engines ready", flush=True)
            print(
                f"[yeetllm] API listening on {self.config.server.host}:"
                f"{self.config.server.port}",
                flush=True,
            )
        else:
            failed = True
            self.state.data["supervisor"]["phase"] = "degraded"
            self.state.touch()
            print(
                "[yeetllm] ERROR: one or more required engines failed; "
                "API readiness is unhealthy and debugging services remain available",
                flush=True,
            )

        await self._monitor_primary()
        return 1 if failed or self._had_failure else 0

    async def _prepare_model(self, index: int) -> None:
        model = self.config.models[index]
        print(f"[yeetllm] preparing {model.id}", flush=True)
        self._set_engine_state(model.id, "preparing")
        self.adapter_paths[model.id] = await asyncio.to_thread(resolve_model_adapters, model)

    async def _launch_primary_engine(self, index: int) -> None:
        model = self.config.models[index]
        record = self.registry.engines[model.id]
        launch = build_engine_launch(
            model,
            port=record.port,
            cluster=self.cluster,
            model_index=index,
            adapter_paths=self.adapter_paths.get(model.id),
        )
        managed = ManagedProcess(
            f"engine:{model.id}", launch.argv, launch.env, run_as_service_user=True
        )
        self.processes[f"engine:{model.id}"] = managed
        print(
            f"[yeetllm] starting {model.id} on local GPUs "
            f"{','.join(map(str, model.gpus))}",
            flush=True,
        )
        print(
            f"[engine:{model.id}] model download/cache directory: "
            f"{PERSISTENT_MODEL_DOWNLOAD_DIR}",
            flush=True,
        )
        self._set_engine_state(model.id, "starting")
        await managed.start()
        self._set_engine_state(model.id, "starting", pid=managed.pid)
        await self._wait_engine_ready(model.id, managed)
        self._set_engine_state(model.id, "ready", pid=managed.pid)
        print(f"[yeetllm] {model.id} ready", flush=True)

    async def _wait_engine_ready(self, engine_id: str, process: ManagedProcess) -> None:
        engine = self.registry.engines[engine_id]
        model = next(item for item in self.config.models if item.id == engine_id)
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + self.config.cluster.startup_timeout_seconds
        next_progress = started + STARTUP_PROGRESS_INTERVAL_SECONDS
        probe_state = "backend not listening"
        timeout = httpx.Timeout(connect=2, read=5, write=5, pool=2)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            while not self.stop_event.is_set():
                if not process.running():
                    raise RuntimeError(
                        f"engine process exited with status {process.returncode} during startup"
                    )
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError(
                        f"engine {engine_id} did not become ready within "
                        f"{self.config.cluster.startup_timeout_seconds:g}s"
                    )
                try:
                    health = await client.get(f"{engine.backend_url}/health")
                    models = await client.get(f"{engine.backend_url}/v1/models")
                    probe_state = f"health={health.status_code} catalog={models.status_code}"
                    if health.status_code == 200 and models.status_code == 200:
                        available = {
                            item.get("id")
                            for item in models.json().get("data", [])
                            if isinstance(item, dict)
                        }
                        if set(engine.model_ids).issubset(available):
                            return
                except httpx.ConnectError:
                    probe_state = "backend not listening"
                except httpx.TimeoutException:
                    probe_state = "backend probe timed out"
                except (httpx.HTTPError, ValueError, TypeError) as exc:
                    probe_state = f"backend probe error ({type(exc).__name__})"
                now = loop.time()
                if now >= next_progress:
                    self._report_startup_progress(
                        engine_id,
                        model.model,
                        process,
                        elapsed_seconds=now - started,
                        probe_state=probe_state,
                    )
                    next_progress = now + STARTUP_PROGRESS_INTERVAL_SECONDS
                await asyncio.sleep(2)
        raise RuntimeError("shutdown requested while engine was starting")

    def _report_startup_progress(
        self,
        engine_id: str,
        model_reference: str,
        process: ManagedProcess,
        *,
        elapsed_seconds: float,
        probe_state: str,
    ) -> None:
        cache_bytes, incomplete_files = model_cache_progress(model_reference)
        cache_message = "cache=not detected"
        if cache_bytes is not None:
            cache_message = f"cache={format_bytes(cache_bytes)}"
            if incomplete_files:
                cache_message += f" incomplete_files={incomplete_files}"
        message = (
            f"elapsed={elapsed_seconds:.0f}s pid={process.pid} "
            f"{probe_state} {cache_message}"
        )
        print(f"[engine:{engine_id}] startup progress: {message}", flush=True)
        record = self.state.data["engines"][engine_id]
        record["progress"] = {
            "elapsed_seconds": round(elapsed_seconds),
            "backend": probe_state,
            "cache_bytes": cache_bytes,
            "incomplete_files": incomplete_files,
        }
        record["updated_at"] = time.time()
        self.state.touch()

    async def _start_router(self) -> None:
        command = [
            sys.executable,
            "-m",
            "yeetllm.router",
            "--state",
            str(self.state.path),
            "--host",
            self.config.server.host,
            "--port",
            str(self.config.server.port),
        ]
        managed = ManagedProcess(
            "router", command, sanitized_environment(), run_as_service_user=True
        )
        self.processes["router"] = managed
        await managed.start()
        self.state.data["router"].update({"state": "running", "pid": managed.pid})
        self.state.touch()
        await self._wait_router_live(managed)

    async def _wait_router_live(self, process: ManagedProcess) -> None:
        deadline = asyncio.get_running_loop().time() + 30
        url = f"http://{self.config.server.host}:{self.config.server.port}/health/live"
        async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
            while asyncio.get_running_loop().time() < deadline:
                if not process.running():
                    raise RuntimeError(f"router exited with status {process.returncode}")
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.25)
        raise TimeoutError("router did not become live within 30 seconds")

    async def _start_ssh(self) -> None:
        launch: SSHLaunch = await asyncio.to_thread(
            prepare_ssh, self.config.ssh, primary=True
        )
        if not launch.enabled or launch.argv is None:
            return
        managed = ManagedProcess("sshd", launch.argv, sanitized_environment())
        self.processes["sshd"] = managed
        await managed.start()
        await asyncio.sleep(0.1)
        if not managed.running():
            raise RuntimeError(f"sshd exited with status {managed.returncode}")
        self.state.data["ssh"].update(
            {
                "enabled": True,
                "state": "running",
                "pid": managed.pid,
                "key_source": launch.key_source,
            }
        )
        self.state.touch()

    async def _monitor_primary(self) -> None:
        while not self.stop_event.is_set():
            degraded = False
            for engine_id in self.registry.engines:
                process = self.processes.get(f"engine:{engine_id}")
                current = self.state.data["engines"][engine_id]["state"]
                if process is not None and not process.running() and current != "failed":
                    self._mark_engine_failed(
                        engine_id,
                        RuntimeError(f"process exited with status {process.returncode}"),
                    )
                if self.state.data["engines"][engine_id]["state"] != "ready":
                    degraded = True
            unhealthy = await self._unhealthy_ready_engines()
            for engine_id, reason in unhealthy.items():
                self._set_engine_state(engine_id, "failed", error=reason)
                print(f"[engine:{engine_id}] ERROR: {reason}", flush=True)
                self._had_failure = True
                degraded = True
            router = self.processes.get("router")
            if router is None or not router.running():
                self.state.data["router"]["state"] = "failed"
                degraded = True
            ssh = self.processes.get("sshd")
            if self.state.data["ssh"]["enabled"] and (ssh is None or not ssh.running()):
                self.state.data["ssh"]["state"] = "failed"
                degraded = True
            if self.coordinator is not None:
                ready_ids = [
                    key
                    for key, value in self.state.data["engines"].items()
                    if value["state"] == "ready"
                ]
                if ready_ids and not self.coordinator.all_workers_running(ready_ids):
                    print("[cluster] ERROR: one or more workers became unavailable", flush=True)
                    for engine_id in ready_ids:
                        self._set_engine_state(
                            engine_id, "failed", error="distributed worker unavailable"
                        )
                    degraded = True
                self._sync_coordinator_desired()
            if degraded:
                self.state.data["supervisor"]["phase"] = "degraded"
            self.state.touch()
            await wait_or_stop(self.stop_event, 2)

    async def _unhealthy_ready_engines(self) -> dict[str, str]:
        ready = [
            engine_id
            for engine_id, value in self.state.data["engines"].items()
            if value["state"] == "ready"
        ]
        if not ready:
            return {}
        timeout = httpx.Timeout(connect=2, read=3, write=3, pool=2)
        unhealthy: dict[str, str] = {}
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            for engine_id in ready:
                try:
                    response = await client.get(
                        f"{self.registry.engines[engine_id].backend_url}/health"
                    )
                    if response.status_code != 200:
                        unhealthy[engine_id] = (
                            f"health endpoint returned HTTP {response.status_code}"
                        )
                except httpx.HTTPError as exc:
                    unhealthy[engine_id] = f"health probe failed: {exc}"
        return unhealthy

    async def _run_worker(self) -> int:
        print(
            f"[cluster] worker rank {self.cluster.node_rank} polling "
            f"{self.cluster.primary_addr}:{self.cluster.control_port}",
            flush=True,
        )
        engine_states: dict[str, str] = {}
        disconnected_since: float | None = None
        connected_once = False
        failed = False
        while not self.stop_event.is_set():
            for identifier, status in list(engine_states.items()):
                process = self.processes.get(f"engine:{identifier}")
                if process is not None and not process.running() and status != "failed":
                    engine_states[identifier] = "failed"
                    self._set_engine_state(
                        identifier,
                        "failed",
                        error=f"headless process exited with status {process.returncode}",
                    )
                    failed = True
            try:
                response = await poll_coordinator(self.cluster, self.config_hash, engine_states)
                disconnected_since = None
                connected_once = True
            except Exception as exc:  # noqa: BLE001 - transient private-network boundary
                disconnect_limit = (
                    60.0
                    if connected_once
                    else self.config.cluster.startup_timeout_seconds
                )
                if disconnected_since is None:
                    disconnected_since = time.monotonic()
                    print(f"[cluster] coordinator unavailable: {exc}", flush=True)
                elif time.monotonic() - disconnected_since >= disconnect_limit:
                    print(
                        "[cluster] ERROR: coordinator unavailable for "
                        f"{disconnect_limit:g} seconds; shutting down",
                        flush=True,
                    )
                    failed = True
                    break
                await wait_or_stop(self.stop_event, 2)
                continue
            if response.get("shutdown"):
                break
            desired = response.get("desired", [])
            self.state.data["cluster"]["desired"] = list(desired)
            removed = [identifier for identifier in engine_states if identifier not in desired]
            for identifier in removed:
                process = self.processes.pop(f"engine:{identifier}", None)
                if process is not None:
                    await process.stop()
                engine_states.pop(identifier, None)
                self._set_engine_state(identifier, "stopped")
                print(f"[cluster] headless engine {identifier} stopped", flush=True)
            new_indices = [
                index
                for index, model in enumerate(self.config.models)
                if model.id in desired and model.id not in engine_states
            ]
            if new_indices:
                results = await asyncio.gather(
                    *(self._launch_worker_engine(index) for index in new_indices),
                    return_exceptions=True,
                )
                for index, result in zip(new_indices, results, strict=True):
                    identifier = self.config.models[index].id
                    if isinstance(result, BaseException):
                        engine_states[identifier] = "failed"
                        self._mark_engine_failed(identifier, result)
                        failed = True
                    else:
                        engine_states[identifier] = "running"
            active = [value for value in engine_states.values()]
            if active and all(value == "running" for value in active):
                self.state.data["supervisor"]["phase"] = "ready"
            elif any(value == "failed" for value in active):
                self.state.data["supervisor"]["phase"] = "degraded"
            self.state.touch()
            await wait_or_stop(self.stop_event, 2)
        return 1 if failed else 0

    async def _launch_worker_engine(self, index: int) -> None:
        model = self.config.models[index]
        self._set_engine_state(model.id, "preparing")
        paths = await asyncio.to_thread(resolve_model_adapters, model)
        record = self.registry.engines[model.id]
        launch = build_engine_launch(
            model,
            port=record.port,
            cluster=self.cluster,
            model_index=index,
            adapter_paths=paths,
        )
        managed = ManagedProcess(
            f"engine:{model.id}", launch.argv, launch.env, run_as_service_user=True
        )
        self.processes[f"engine:{model.id}"] = managed
        self._set_engine_state(model.id, "starting")
        await managed.start()
        await asyncio.sleep(0.25)
        if not managed.running():
            raise RuntimeError(f"headless process exited with status {managed.returncode}")
        self._set_engine_state(model.id, "running", pid=managed.pid)
        print(f"[cluster] headless engine {model.id} running", flush=True)

    async def _heartbeat(self) -> None:
        while not self.stop_event.is_set():
            self.state.touch()
            await wait_or_stop(self.stop_event, 2)

    async def _shutdown(self) -> None:
        self.stop_event.set()
        self.state.data.setdefault("supervisor", {})["phase"] = "stopping"
        self.state.touch()
        if self.coordinator is not None:
            self.coordinator.shutting_down = True
            await asyncio.sleep(3)
            await self.coordinator.close()
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
        children = list(reversed(self.processes.values()))
        results = await asyncio.gather(
            *(process.stop() for process in children), return_exceptions=True
        )
        for process, result in zip(children, results, strict=True):
            if isinstance(result, BaseException):
                print(f"[{process.name}] ERROR during shutdown: {result}", flush=True)
        self.state.data["supervisor"]["phase"] = "stopped"
        self.state.touch()
        print("[yeetllm] shutdown complete", flush=True)

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for received in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(received, self.stop_event.set)

    def _set_engine_state(
        self,
        identifier: str,
        state: str,
        *,
        pid: int | None = None,
        error: str | None = None,
    ) -> None:
        record = self.state.data["engines"][identifier]
        record["state"] = state
        if pid is not None:
            record["pid"] = pid
        record["error"] = error
        record["updated_at"] = time.time()
        self.state.touch()

    def _mark_engine_failed(self, identifier: str, error: BaseException) -> None:
        message = str(error) or type(error).__name__
        self._set_engine_state(identifier, "failed", error=message)
        self._had_failure = True
        print(f"[engine:{identifier}] ERROR: {message}", flush=True)

    def _sync_coordinator_desired(self) -> None:
        if self.coordinator is None:
            return
        desired = [
            identifier
            for identifier, value in self.state.data["engines"].items()
            if value["state"] in {"preparing", "starting", "ready"}
        ]
        self.coordinator.set_desired(desired)

    def _reserved_listener_ports(self) -> dict[int, str]:
        ports = {
            self.config.server.port: "router",
            self.config.ssh.port: "sshd",
        }
        ports.update(
            {record.port: f"engine {record.id}" for record in self.registry.engines.values()}
        )
        return ports

    def _print_summary(self) -> None:
        role = "primary" if self.cluster.primary else f"worker (rank {self.cluster.node_rank})"
        print("YeetLLM", flush=True)
        print(f"Node: {role}", flush=True)
        print(f"GPUs: {self.gpu_count}", flush=True)
        if self.cluster.primary:
            print(f"API: {self.config.server.host}:{self.config.server.port}", flush=True)
        if self.cluster.enabled:
            print(
                f"[cluster] rank={self.cluster.node_rank} nodes={self.cluster.node_count} "
                f"world_size={self.cluster.world_size} master={self.cluster.primary_addr} "
                f"interface={self.cluster.interface} backend=mp",
                flush=True,
            )
        print("Models:", flush=True)
        for model in self.config.models:
            adapters = ", ".join(adapter.id for adapter in model.lora.adapters) or "none"
            print(
                f"  {model.id}: {model.model}; GPUs={','.join(map(str, model.gpus))}; "
                f"TP={model.tensor_parallel_size}; PP={model.pipeline_parallel_size}; "
                f"LoRA={adapters}",
                flush=True,
            )


def chunked(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


async def wait_or_stop(event: asyncio.Event, seconds: float) -> None:
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(event.wait(), timeout=seconds)


def model_cache_progress(model_reference: str) -> tuple[int | None, int]:
    """Return locally downloaded blob bytes without contacting Hugging Face."""

    candidate = Path(model_reference)
    if candidate.exists() or "/" not in model_reference:
        return None, 0
    hub = Path(PERSISTENT_MODEL_DOWNLOAD_DIR)
    blobs = hub / f"models--{model_reference.replace('/', '--')}" / "blobs"
    if not blobs.is_dir():
        return None, 0
    total = 0
    incomplete = 0
    try:
        paths = list(blobs.iterdir())
    except OSError:
        return None, 0
    for path in paths:
        try:
            if not path.is_file():
                continue
            total += path.stat().st_size
            if path.name.endswith(".incomplete"):
                incomplete += 1
        except OSError:
            continue
    return total, incomplete


def format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f}{unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def sanitized_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "YEETLLM_CONFIG_URL",
        "YEETLLM_CONFIG_SHA256",
        "YEETLLM_SSH_AUTHORIZED_KEYS",
        "SSH_PUBLIC_KEY",
        "PUBLIC_KEY",
    ):
        environment.pop(name, None)
    return environment
