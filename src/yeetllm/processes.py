from __future__ import annotations

import asyncio
import contextlib
import os
import pwd
import signal
from dataclasses import dataclass, field


@dataclass
class ManagedProcess:
    name: str
    argv: list[str]
    env: dict[str, str] | None = None
    run_as_service_user: bool = False
    process: asyncio.subprocess.Process | None = None
    readers: list[asyncio.Task[None]] = field(default_factory=list)

    async def start(self) -> None:
        argv = service_argv(self.argv) if self.run_as_service_user else self.argv
        environment = dict(os.environ if self.env is None else self.env)
        if self.run_as_service_user:
            environment["HOME"] = "/home/vllm"
            environment["USER"] = "vllm"
        self.process = await asyncio.create_subprocess_exec(
            *argv,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=1024 * 1024,
        )
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self.readers = [
            asyncio.create_task(self._read(self.process.stdout)),
            asyncio.create_task(self._read(self.process.stderr)),
        ]

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    @property
    def returncode(self) -> int | None:
        return self.process.returncode if self.process is not None else None

    def running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def stop(self, grace_seconds: float = 20.0) -> None:
        if self.process is None or self.process.returncode is not None:
            await self._finish_readers()
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(self.process.wait(), timeout=grace_seconds)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            await self.process.wait()
        await self._finish_readers()

    async def _read(self, stream: asyncio.StreamReader) -> None:
        while line := await stream.readline():
            print(f"[{self.name}] {line.decode(errors='replace').rstrip()}", flush=True)

    async def _finish_readers(self) -> None:
        if self.readers:
            await asyncio.gather(*self.readers, return_exceptions=True)


def service_argv(argv: list[str]) -> list[str]:
    if os.geteuid() != 0:
        return argv
    # The pinned official vLLM image provides this non-root runtime account.
    account = pwd.getpwnam("vllm")
    return [
        "/usr/bin/setpriv",
        f"--reuid={account.pw_uid}",
        f"--regid={account.pw_gid}",
        "--init-groups",
        "--no-new-privs",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--",
        *argv,
    ]
