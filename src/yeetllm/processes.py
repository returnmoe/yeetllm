from __future__ import annotations

import asyncio
import contextlib
import os
import pwd
import signal
import subprocess
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
        # tqdm and Hugging Face download progress use carriage returns rather
        # than newlines. StreamReader.readline() can therefore accumulate past
        # its configured limit, terminate this reader task, and hide all later
        # vLLM output. Read bounded chunks and frame both line conventions.
        pending = bytearray()
        while chunk := await stream.read(64 * 1024):
            pending.extend(chunk)
            self._emit_complete_lines(pending)
            while len(pending) > 64 * 1024:
                self._emit_log(bytes(pending[: 64 * 1024]))
                del pending[: 64 * 1024]
        if pending:
            self._emit_log(bytes(pending))

    def _emit_complete_lines(self, pending: bytearray) -> None:
        while True:
            newline = pending.find(b"\n")
            carriage_return = pending.find(b"\r")
            positions = [position for position in (newline, carriage_return) if position >= 0]
            if not positions:
                return
            boundary = min(positions)
            line = bytes(pending[:boundary])
            delimiter = pending[boundary]
            del pending[: boundary + 1]
            if delimiter == ord("\r") and pending[:1] == b"\n":
                del pending[:1]
            if line:
                self._emit_log(line)

    def _emit_log(self, line: bytes) -> None:
        print(f"[{self.name}] {line.decode(errors='replace')}", flush=True)

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


def service_path_writable(path: os.PathLike[str] | str) -> bool:
    """Check path write access as the actual engine service account."""

    result = subprocess.run(  # noqa: S603 - fixed argv, never a shell
        service_argv(["/usr/bin/test", "-w", os.fspath(path)]),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0
