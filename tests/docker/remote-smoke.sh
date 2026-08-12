#!/usr/bin/env bash
set -Eeuo pipefail

image="${1:?usage: remote-smoke.sh IMAGE}"
config_file="$(mktemp)"
trap 'rm -f "${config_file}"' EXIT

docker buildx imagetools inspect "${image}" \
  --format '{{json (index .Image "linux/amd64").Config}}' > "${config_file}"

python3 - "${config_file}" "${image}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
image = sys.argv[2]
config = json.loads(config_path.read_text(encoding="utf-8"))

expected_entrypoint = ["/usr/bin/tini", "--", "/usr/local/bin/yeetllm-entrypoint"]
expected_cmd = ["yeetllm", "serve"]
if config.get("Entrypoint") != expected_entrypoint:
    raise SystemExit(f"unexpected entrypoint: {config.get('Entrypoint')!r}")
if config.get("Cmd") != expected_cmd:
    raise SystemExit(f"unexpected command: {config.get('Cmd')!r}")
if config.get("ExposedPorts") != {"22/tcp": {}}:
    raise SystemExit(f"unexpected exposed ports: {config.get('ExposedPorts')!r}")

environment = set(config.get("Env", []))
required_environment = {
    "PYTHONPATH=/opt/yeetllm",
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING=0",
    "YEETLLM_EXPECTED_CUDA=13.0",
}
missing = required_environment - environment
if missing:
    raise SystemExit(f"missing hardened environment settings: {sorted(missing)!r}")

labels = config.get("Labels", {})
if labels.get("ai.yeetllm.vllm.provenance") != "official-vllm-image":
    raise SystemExit("image does not declare official vLLM base provenance")

print(f"remote image configuration passed: {image}")
PY
