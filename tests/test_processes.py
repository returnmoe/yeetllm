from __future__ import annotations

import sys

import pytest

from yeetllm.processes import ManagedProcess
from yeetllm.supervisor import sanitized_environment


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
