from __future__ import annotations

import sys

import pytest

from yeetllm.processes import ManagedProcess


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
