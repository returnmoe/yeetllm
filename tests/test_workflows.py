from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_workflow(name: str) -> dict[str, Any]:
    path = Path(__file__).parents[1] / ".github" / "workflows" / name
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_gpu_workflow_does_not_interpolate_dispatch_inputs_into_shell() -> None:
    workflow = load_workflow("gpu-integration.yml")
    job = workflow["jobs"]["gpu"]
    assert job["environment"] == "gpu-qualification"
    assert "yeetllm-ephemeral" in job["runs-on"]
    assert job["if"] == "github.ref == 'refs/heads/development'"

    checkout = job["steps"][0]
    assert checkout["with"]["persist-credentials"] is False

    run_steps = [step for step in job["steps"] if "run" in step]
    assert run_steps
    assert all("${{ inputs." not in step["run"] for step in run_steps)

    profile = run_steps[-1]
    assert profile["env"]["YEETLLM_IMAGE_INPUT"] == "${{ inputs.image }}"
    assert profile["env"]["YEETLLM_CONFIG_INPUT"] == "${{ inputs.config }}"
    assert "^ghcr\\.io/returnmoe/yeetllm" in profile["run"]
    assert "^/opt/yeetllm-gpu-configs/" in profile["run"]
    assert "realpath -e" in profile["run"]
