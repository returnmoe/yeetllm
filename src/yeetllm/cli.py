from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from yeetllm import __version__
from yeetllm.cluster import detect_cluster, validate_cluster_ports
from yeetllm.config import (
    RuntimeValidationError,
    config_path,
    discover_gpu_count,
    installed_quantization_methods,
    load_config,
    local_adapter_rank,
    validate_runtime,
)
from yeetllm.registry import build_registry
from yeetllm.state import DEFAULT_STATE_PATH, read_state
from yeetllm.supervisor import Supervisor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yeetllm", description="RunPod-ready vLLM appliance")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    serve = subcommands.add_parser("serve", help="start the supervisor and configured models")
    serve.add_argument("--config", type=Path)
    serve.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)

    validate = subcommands.add_parser("validate", help="validate without downloading models")
    validate.add_argument("--config", type=Path)
    validate.add_argument(
        "--gpu-count",
        type=int,
        help="validate against this GPU count instead of running nvidia-smi",
    )
    validate.add_argument(
        "--skip-interface-check",
        action="store_true",
        help="skip the RunPod ens1 address check (intended for CI only)",
    )

    status = subcommands.add_parser("status", help="show the runtime registry and health")
    status.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    status.add_argument("--json", action="store_true")
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "serve":
            config = load_config(args.config)
            supervisor = Supervisor(config, state_path=args.state)
            return asyncio.run(supervisor.run())
        if args.command == "validate":
            return validate_command(args.config, args.gpu_count, args.skip_interface_check)
        if args.command == "status":
            return status_command(args.state, args.json)
    except (RuntimeValidationError, ValidationError, ValueError, OSError) as exc:
        print(f"[yeetllm] ERROR: {exc}", file=sys.stderr)
        return 2
    return 2


def validate_command(path: Path | None, gpu_count: int | None, skip_interface: bool) -> int:
    config = load_config(path)
    cluster = detect_cluster(config.cluster, verify_interface=not skip_interface)
    discovered = discover_gpu_count() if gpu_count is None else gpu_count
    if discovered < 1:
        raise RuntimeValidationError("GPU count must be at least one")
    if cluster.enabled and discovered != cluster.trainers_per_node:
        raise RuntimeValidationError(
            f"NUM_TRAINERS={cluster.trainers_per_node}, but GPU count is {discovered}"
        )
    validate_cluster_ports(
        cluster,
        len(config.models),
        reserved_ports={
            config.server.port: "router",
            config.ssh.port: "sshd",
            **{
                record.port: f"engine {record.id}"
                for record in build_registry(config).engines.values()
            },
        },
    )
    warnings = validate_runtime(
        config,
        gpu_count=discovered,
        node_count=cluster.node_count,
        quantization_methods=installed_quantization_methods(),
        rank_resolver=local_adapter_rank,
    )
    registry = build_registry(config)
    selected_path = path or config_path()
    print(f"[yeetllm] configuration valid: {selected_path}")
    print(
        f"[yeetllm] {len(registry.engines)} engines, {len(registry.models)} selectable models, "
        f"{discovered} local GPUs"
    )
    for warning in warnings:
        print(f"[yeetllm] WARNING: {warning}")
    return 0


def status_command(path: Path, as_json: bool) -> int:
    state = read_state(path)
    if as_json:
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    supervisor = state.get("supervisor", {})
    heartbeat = float(supervisor.get("heartbeat", 0))
    age = max(0.0, time.time() - heartbeat)
    node = state.get("node", {})
    print(
        f"YeetLLM: {supervisor.get('phase', 'unknown')} "
        f"({node.get('role', 'unknown')}, heartbeat {age:.1f}s ago)"
    )
    registry = state.get("registry", {}).get("models", {})
    engines: dict[str, dict[str, Any]] = state.get("engines", {})
    for identifier, record in registry.items():
        engine_id = record.get("engine_id", "unknown")
        health = engines.get(engine_id, {}).get("state", "unknown")
        kind = record.get("kind", "unknown")
        parent = f", parent={record.get('parent')}" if record.get("parent") else ""
        print(f"  {identifier}: {health} ({kind}, engine={engine_id}{parent})")
    ssh = state.get("ssh", {})
    print(f"SSH: {ssh.get('state', 'unknown')}")
    return 0 if supervisor.get("phase") == "ready" else 1


def main() -> None:
    raise SystemExit(run_cli())
