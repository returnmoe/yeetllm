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


def test_ci_does_not_materialize_the_full_cuda_image() -> None:
    workflow = load_workflow("ci.yml")
    steps = workflow["jobs"]["image-graph"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)

    assert "docker buildx build --check ." in commands
    assert "--target runtime-tools" in commands
    assert "image.output=type=cacheonly" not in commands


def test_image_publication_reclaims_runner_disk_before_buildx() -> None:
    for workflow_name in ("development.yml", "release.yml"):
        workflow = load_workflow(workflow_name)
        job_name = "publish" if workflow_name == "development.yml" else "build"
        steps = workflow["jobs"][job_name]["steps"]
        cleanup_index = next(
            index
            for index, step in enumerate(steps)
            if "free-build-disk.sh 30" in step.get("run", "")
        )
        buildx_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("uses", "").startswith("docker/setup-buildx-action@")
        )

        assert cleanup_index < buildx_index
        assert any(step.get("if") == "always()" for step in steps)


def test_remote_smoke_accepts_single_and_multi_platform_metadata() -> None:
    script = (
        Path(__file__).parents[1] / "tests" / "docker" / "remote-smoke.sh"
    ).read_text(encoding="utf-8")

    assert "image_metadata.get(\"config\")" in script
    assert "image_metadata.get(\"linux/amd64\")" in script
