from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from yeetllm.state import DEFAULT_STATE_PATH, read_state


def check(path: Path = DEFAULT_STATE_PATH) -> list[str]:
    failures: list[str] = []
    try:
        state = read_state(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"state unavailable: {exc}"]

    supervisor = state.get("supervisor", {})
    heartbeat = float(supervisor.get("heartbeat", 0))
    if time.time() - heartbeat > 15:
        failures.append("supervisor heartbeat is stale")
    if supervisor.get("phase") != "ready":
        failures.append(f"supervisor phase is {supervisor.get('phase', 'unknown')}")
    require_process(supervisor.get("pid"), "supervisor", failures)

    node = state.get("node", {})
    if node.get("role") == "worker":
        desired = set(state.get("cluster", {}).get("desired", []))
        engines = state.get("engines", {})
        for identifier in desired:
            record = engines.get(identifier, {})
            if record.get("state") != "running":
                failures.append(
                    f"worker engine {identifier} is {record.get('state', 'unknown')}"
                )
            require_process(record.get("pid"), f"worker engine {identifier}", failures)
        return failures

    router = state.get("router", {})
    if router.get("state") != "running":
        failures.append(f"router is {router.get('state', 'unknown')}")
    require_process(router.get("pid"), "router", failures)
    host = router.get("host", "127.0.0.1")
    port = router.get("port", 8000)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://{host}:{port}/health/ready", timeout=5) as response:
            if response.status != 200:
                failures.append(f"router readiness returned HTTP {response.status}")
    except (OSError, urllib.error.URLError) as exc:
        failures.append(f"router readiness failed: {exc}")

    for identifier, record in state.get("engines", {}).items():
        if record.get("state") != "ready":
            failures.append(f"engine {identifier} is {record.get('state', 'unknown')}")
        require_process(record.get("pid"), f"engine {identifier}", failures)

    ssh = state.get("ssh", {})
    if ssh.get("enabled"):
        if ssh.get("state") != "running":
            failures.append(f"sshd is {ssh.get('state', 'unknown')}")
        require_process(ssh.get("pid"), "sshd", failures)
        result = subprocess.run(  # noqa: S603 - fixed sshd validation command
            [
                "/usr/sbin/sshd",
                "-t",
                "-f",
                "/etc/ssh/sshd_config",
                "-o",
                f"Port={ssh.get('port', 22)}",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failures.append(f"sshd configuration invalid: {result.stderr.strip()}")
    return failures


def require_process(pid: Any, name: str, failures: list[str]) -> None:
    if not isinstance(pid, int) or pid < 1 or not Path(f"/proc/{pid}").exists():
        failures.append(f"{name} process is not running")


def main() -> None:
    path = Path(os.environ.get("YEETLLM_STATE", str(DEFAULT_STATE_PATH)))
    failures = check(path)
    if failures:
        for failure in failures:
            print(f"[healthcheck] {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("[healthcheck] ready")


if __name__ == "__main__":
    main()
