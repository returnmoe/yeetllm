from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

import yeetllm.supervisor
from yeetllm.processes import ManagedProcess
from yeetllm.supervisor import format_bytes, model_cache_progress, sanitized_environment


@pytest.mark.asyncio
async def test_managed_process_terminates_its_process_group() -> None:
    managed = ManagedProcess(
        "test",
        [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(60)"],
    )
    await managed.start()
    assert managed.running()
    await managed.stop(grace_seconds=2)
    assert not managed.running()


@pytest.mark.asyncio
async def test_managed_process_forwards_carriage_return_logs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    managed = ManagedProcess(
        "engine:test",
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('download 1%\\rdownload 2%\\n'); sys.stdout.flush()",
        ],
    )
    await managed.start()
    assert managed.process is not None
    await asyncio.wait_for(managed.process.wait(), timeout=2)
    await managed.stop()

    output = capsys.readouterr().out
    assert "[engine:test] download 1%" in output
    assert "[engine:test] download 2%" in output


def test_router_and_sshd_environment_omits_configuration_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "YEETLLM_CONFIG_URL", "https://config.example/config.yaml?token=sensitive"
    )
    monkeypatch.setenv("YEETLLM_CONFIG_SHA256", "a" * 64)

    environment = sanitized_environment()

    assert "YEETLLM_CONFIG_URL" not in environment
    assert "YEETLLM_CONFIG_SHA256" not in environment


def test_model_cache_progress_counts_completed_and_partial_blobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = tmp_path / "hub"
    blobs = hub / "models--org--model" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "complete").write_bytes(b"a" * 10)
    (blobs / "partial.incomplete").write_bytes(b"b" * 7)
    monkeypatch.setattr(yeetllm.supervisor, "PERSISTENT_MODEL_DOWNLOAD_DIR", str(hub))

    assert model_cache_progress("org/model") == (17, 1)
    assert format_bytes(1024) == "1.0KiB"
