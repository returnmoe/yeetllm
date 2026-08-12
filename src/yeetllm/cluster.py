from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yeetllm.config import ClusterConfig, RuntimeValidationError

RUNPOD_CLUSTER_KEYS = {
    "PRIMARY_ADDR",
    "MASTER_ADDR",
    "PRIMARY_PORT",
    "MASTER_PORT",
    "NODE_ADDR",
    "NODE_RANK",
    "NUM_NODES",
    "NUM_TRAINERS",
    "HOST_NODE_ADDR",
    "WORLD_SIZE",
}


@dataclass(frozen=True)
class ClusterInfo:
    enabled: bool
    role: str
    node_rank: int = 0
    node_count: int = 1
    trainers_per_node: int = 0
    world_size: int = 0
    primary_addr: str = "127.0.0.1"
    node_addr: str = "127.0.0.1"
    control_port: int = 0
    rendezvous_port_base: int = 0
    interface: str = ""

    @property
    def primary(self) -> bool:
        return self.node_rank == 0


def detect_cluster(
    config: ClusterConfig,
    env: dict[str, str] | None = None,
    *,
    verify_interface: bool = True,
) -> ClusterInfo:
    source = dict(os.environ if env is None else env)
    present = RUNPOD_CLUSTER_KEYS.intersection(source)
    if config.mode == "off":
        return ClusterInfo(enabled=False, role="primary")
    if not present:
        if config.mode == "runpod":
            raise RuntimeValidationError(
                "cluster.mode=runpod but RunPod cluster variables are absent"
            )
        return ClusterInfo(enabled=False, role="primary")

    missing = sorted(RUNPOD_CLUSTER_KEYS.difference(source))
    if missing:
        raise RuntimeValidationError(
            "partial RunPod cluster environment; missing: " + ", ".join(missing)
        )
    if source["PRIMARY_ADDR"] != source["MASTER_ADDR"]:
        raise RuntimeValidationError("PRIMARY_ADDR and MASTER_ADDR disagree")
    if source["PRIMARY_PORT"] != source["MASTER_PORT"]:
        raise RuntimeValidationError("PRIMARY_PORT and MASTER_PORT disagree")

    primary_port = parse_positive_int(source, "PRIMARY_PORT", maximum=65535)
    node_rank = parse_nonnegative_int(source, "NODE_RANK")
    node_count = parse_positive_int(source, "NUM_NODES")
    trainers = parse_positive_int(source, "NUM_TRAINERS")
    world_size = parse_positive_int(source, "WORLD_SIZE")
    if node_rank >= node_count:
        raise RuntimeValidationError("NODE_RANK must be less than NUM_NODES")
    if node_rank == 0 and source["PRIMARY_ADDR"] != source["NODE_ADDR"]:
        raise RuntimeValidationError("rank 0 PRIMARY_ADDR must equal NODE_ADDR")
    if world_size != node_count * trainers:
        raise RuntimeValidationError("WORLD_SIZE must equal NUM_NODES x NUM_TRAINERS")
    expected_host = f"{source['PRIMARY_ADDR']}:{primary_port}"
    if source["HOST_NODE_ADDR"] != expected_host:
        raise RuntimeValidationError(f"HOST_NODE_ADDR must equal {expected_host}")
    if node_count == 1 and config.mode == "auto":
        return ClusterInfo(enabled=False, role="primary")
    if config.nccl_socket_ifname == "eth0":
        raise RuntimeValidationError(
            "eth0 is the management interface; use ens1 for cluster traffic"
        )
    if verify_interface:
        verify_address_on_interface(source["NODE_ADDR"], config.nccl_socket_ifname)

    control_port = config.control_port or primary_port
    rendezvous = config.rendezvous_port_base or (primary_port + 1)
    if rendezvous > 65535:
        raise RuntimeValidationError("default cluster rendezvous port exceeds 65535")
    return ClusterInfo(
        enabled=True,
        role="primary" if node_rank == 0 else "worker",
        node_rank=node_rank,
        node_count=node_count,
        trainers_per_node=trainers,
        world_size=world_size,
        primary_addr=source["PRIMARY_ADDR"],
        node_addr=source["NODE_ADDR"],
        control_port=control_port,
        rendezvous_port_base=rendezvous,
        interface=config.nccl_socket_ifname,
    )


def validate_cluster_ports(
    info: ClusterInfo,
    model_count: int,
    *,
    reserved_ports: dict[int, str] | None = None,
) -> None:
    if not info.enabled:
        return
    last = info.rendezvous_port_base + model_count - 1
    if last > 65535:
        raise RuntimeValidationError("cluster rendezvous port range exceeds 65535")
    if info.control_port in range(info.rendezvous_port_base, last + 1):
        raise RuntimeValidationError("cluster control port overlaps an engine rendezvous port")
    claims = dict(reserved_ports or {})
    cluster_ports = {info.control_port: "cluster control"}
    cluster_ports.update(
        {
            info.rendezvous_port_base + index: f"cluster rendezvous for engine {index}"
            for index in range(model_count)
        }
    )
    for port, owner in cluster_ports.items():
        previous = claims.get(port)
        if previous is not None:
            raise RuntimeValidationError(
                f"port {port} is assigned to both {previous} and {owner}"
            )


def cluster_environment(info: ClusterInfo) -> dict[str, str]:
    if not info.enabled:
        return {}
    return {
        "NCCL_SOCKET_IFNAME": info.interface,
        "GLOO_SOCKET_IFNAME": info.interface,
        "VLLM_HOST_IP": info.node_addr,
    }


def verify_address_on_interface(address: str, interface: str) -> None:
    path = Path(f"/sys/class/net/{interface}")
    if not path.exists():
        raise RuntimeValidationError(f"cluster interface {interface!r} does not exist")
    import subprocess

    try:
        result = subprocess.run(  # noqa: S603 - fixed executable and validated interface
            ["/usr/bin/ip", "-j", "address", "show", "dev", interface],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        records = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeValidationError(
            f"could not inspect cluster interface {interface}: {exc}"
        ) from exc
    addresses = {
        entry.get("local")
        for record in records
        for entry in record.get("addr_info", [])
        if isinstance(entry, dict)
    }
    if address not in addresses:
        raise RuntimeValidationError(f"NODE_ADDR {address} is not assigned to {interface}")


def parse_positive_int(env: dict[str, str], name: str, maximum: int | None = None) -> int:
    value = parse_nonnegative_int(env, name)
    if value < 1 or (maximum is not None and value > maximum):
        suffix = f" and <= {maximum}" if maximum else ""
        raise RuntimeValidationError(f"{name} must be >= 1{suffix}")
    return value


def parse_nonnegative_int(env: dict[str, str], name: str) -> int:
    try:
        value = int(env[name])
    except ValueError as exc:
        raise RuntimeValidationError(f"{name} must be an integer") from exc
    if value < 0:
        raise RuntimeValidationError(f"{name} must be non-negative")
    return value


@dataclass
class WorkerReport:
    rank: int
    last_seen: float
    engines: dict[str, str] = field(default_factory=dict)


class ClusterCoordinator:
    """Tiny private control plane; it never accepts commands or arbitrary argv."""

    def __init__(
        self,
        info: ClusterInfo,
        config_hash: str,
        allowed_engines: set[str],
    ) -> None:
        self.info = info
        self.config_hash = config_hash
        self.allowed_engines = allowed_engines
        self.desired: list[str] = []
        self.generation = 0
        self.shutting_down = False
        self.reports: dict[int, WorkerReport] = {}
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle, self.info.primary_addr, self.info.control_port, limit=64 * 1024
        )

    async def close(self) -> None:
        self.shutting_down = True
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def set_desired(self, engine_ids: list[str]) -> None:
        if not set(engine_ids).issubset(self.allowed_engines):
            raise ValueError("coordinator desired set contains an unknown engine")
        selected = list(engine_ids)
        if selected != self.desired:
            self.desired = selected
            self.generation += 1

    def all_workers_running(self, engine_ids: list[str], stale_after: float = 15.0) -> bool:
        now = time.time()
        expected = set(range(1, self.info.node_count))
        if set(self.reports) != expected:
            return False
        return all(
            now - report.last_seen <= stale_after
            and all(report.engines.get(engine_id) == "running" for engine_id in engine_ids)
            for report in self.reports.values()
        )

    async def wait_workers(self, engine_ids: list[str], timeout_seconds: float) -> None:
        if self.info.node_count <= 1:
            return
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while not self.all_workers_running(engine_ids):
            failed = {
                rank: engine_id
                for rank, report in self.reports.items()
                for engine_id in engine_ids
                if report.engines.get(engine_id) == "failed"
            }
            if failed:
                detail = ", ".join(
                    f"rank {rank}: {engine_id}" for rank, engine_id in sorted(failed.items())
                )
                raise RuntimeError(f"distributed workers reported failed engines ({detail})")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"workers did not start engines within {timeout_seconds:g}s"
                )
            await asyncio.sleep(1)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=10)
            if len(raw) > 64 * 1024:
                raise ValueError("cluster message too large")
            request = json.loads(raw)
            response = self._process(request)
        except Exception as exc:  # noqa: BLE001 - protocol boundary
            response = {"ok": False, "error": str(exc)}
        writer.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    def _process(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict) or request.get("op") != "poll":
            raise ValueError("unsupported cluster operation")
        if request.get("config_hash") != self.config_hash:
            raise ValueError("cluster configuration hash mismatch")
        rank = request.get("rank")
        if not isinstance(rank, int) or rank <= 0 or rank >= self.info.node_count:
            raise ValueError("invalid worker rank")
        engines = request.get("engines")
        if not isinstance(engines, dict) or not set(engines).issubset(self.allowed_engines):
            raise ValueError("worker reported an unknown engine")
        if not all(value in {"starting", "running", "failed"} for value in engines.values()):
            raise ValueError("invalid worker engine state")
        self.reports[rank] = WorkerReport(rank, time.time(), dict(engines))
        return {
            "ok": True,
            "generation": self.generation,
            "desired": self.desired,
            "shutdown": self.shutting_down,
        }


async def poll_coordinator(
    info: ClusterInfo,
    config_hash: str,
    engines: dict[str, str],
) -> dict[str, Any]:
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(info.primary_addr, info.control_port), timeout=10
    )
    payload = {
        "op": "poll",
        "rank": info.node_rank,
        "config_hash": config_hash,
        "engines": engines,
    }
    writer.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
    await writer.drain()
    raw = await asyncio.wait_for(reader.readline(), timeout=10)
    writer.close()
    await writer.wait_closed()
    response = json.loads(raw)
    if not isinstance(response, dict):
        raise RuntimeError("cluster coordinator returned a non-object response")
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "cluster coordinator rejected worker"))
    return dict(response)
