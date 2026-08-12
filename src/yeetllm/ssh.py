from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from yeetllm.config import RuntimeValidationError, SSHConfig

RUNTIME_DIR = Path("/run/yeetllm")
AUTHORIZED_KEYS = RUNTIME_DIR / "authorized_keys"
HOST_KEY_DIR = RUNTIME_DIR / "ssh"
SSHD_CONFIG = Path("/etc/ssh/sshd_config")


@dataclass(frozen=True)
class SSHLaunch:
    enabled: bool
    argv: list[str] | None = None
    key_source: str | None = None


def select_authorized_keys(
    env: Mapping[str, str] | None = None,
    mounted_path: Path = Path("/root/.ssh/authorized_keys"),
) -> tuple[str, str] | None:
    source = os.environ if env is None else env
    candidates: list[tuple[str, str | None]] = [
        ("YEETLLM_SSH_AUTHORIZED_KEYS", source.get("YEETLLM_SSH_AUTHORIZED_KEYS")),
        ("SSH_PUBLIC_KEY", source.get("SSH_PUBLIC_KEY")),
    ]
    try:
        mounted = mounted_path.read_text(encoding="utf-8") if mounted_path.is_file() else None
    except OSError as exc:
        raise RuntimeValidationError(f"could not read {mounted_path}: {exc}") from exc
    candidates.extend(
        [
            (str(mounted_path), mounted),
            ("PUBLIC_KEY", source.get("PUBLIC_KEY")),
        ]
    )
    for name, value in candidates:
        if value is not None and value.strip():
            return name, value.strip() + "\n"
    return None


def prepare_ssh(
    config: SSHConfig,
    *,
    primary: bool,
    env: Mapping[str, str] | None = None,
    runtime_dir: Path = RUNTIME_DIR,
    sshd_config: Path = SSHD_CONFIG,
    mounted_path: Path = Path("/root/.ssh/authorized_keys"),
) -> SSHLaunch:
    if not primary or config.enable is False:
        return SSHLaunch(enabled=False)
    selected = select_authorized_keys(env, mounted_path)
    if selected is None:
        if config.enable is True:
            raise RuntimeValidationError(
                "SSH is explicitly enabled but no valid authorized public-key source is present"
            )
        if is_runpod(env):
            print(
                "[sshd] RunPod detected, but no injected public key was found; "
                "SSH remains disabled (check startSsh/account key timing)"
            )
        else:
            print("[sshd] no public key detected; SSH remains disabled")
        return SSHLaunch(enabled=False)

    source_name, key_text = selected
    if "PRIVATE KEY-----" in key_text:
        if config.enable is True:
            raise RuntimeValidationError(
                f"authorized key source {source_name} contains private-key material"
            )
        print(
            f"[sshd] authorized-key source {source_name} is not a public key; "
            "SSH remains disabled",
            flush=True,
        )
        return SSHLaunch(enabled=False)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.chmod(0o755)
    authorized_keys = runtime_dir / "authorized_keys"
    write_private_text(authorized_keys, key_text)
    try:
        validate_key_file(authorized_keys, source_name)
    except RuntimeValidationError:
        authorized_keys.unlink(missing_ok=True)
        if config.enable is True:
            raise
        print(
            f"[sshd] authorized-key source {source_name} is invalid; SSH remains disabled",
            flush=True,
        )
        return SSHLaunch(enabled=False)

    host_dir = runtime_dir / "ssh"
    host_dir.mkdir(mode=0o700, exist_ok=True)
    host_dir.chmod(0o700)
    host_keys = [
        ("ed25519", host_dir / "ssh_host_ed25519_key", []),
        ("rsa", host_dir / "ssh_host_rsa_key", ["-b", "3072"]),
    ]
    for key_type, path, extra in host_keys:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".pub").unlink(missing_ok=True)
        subprocess.run(  # noqa: S603 - fixed executable with argv, never a shell
            [
                "/usr/bin/ssh-keygen",
                "-q",
                "-t",
                key_type,
                *extra,
                "-N",
                "",
                "-f",
                str(path),
            ],
            check=True,
        )
        public = path.with_suffix(path.suffix + ".pub")
        public_text = public.read_text(encoding="utf-8").strip()
        fingerprint = subprocess.run(  # noqa: S603 - fixed executable and argv
            ["/usr/bin/ssh-keygen", "-E", "sha256", "-lf", str(public)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        print(f"[sshd] host public key: {public_text}", flush=True)
        print(f"[sshd] host fingerprint: {fingerprint}", flush=True)

    argv = [
        "/usr/sbin/sshd",
        "-D",
        "-e",
        "-f",
        str(sshd_config),
        "-o",
        f"Port={config.port}",
    ]
    subprocess.run([*argv[:1], "-t", *argv[3:]], check=True)  # noqa: S603
    print(
        f"[sshd] authorized key detected from {source_name}; SSH enabled on port {config.port}",
        flush=True,
    )
    return SSHLaunch(enabled=True, argv=argv, key_source=source_name)


def validate_key_file(path: Path, source_name: str) -> None:
    result = subprocess.run(  # noqa: S603 - fixed executable and argv
        ["/usr/bin/ssh-keygen", "-l", "-f", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeValidationError(
            f"authorized key source {source_name} is invalid: {result.stderr.strip()}"
        )
    expected = sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    actual = len([line for line in result.stdout.splitlines() if line.strip()])
    if expected < 1 or actual != expected:
        raise RuntimeValidationError(
            f"authorized key source {source_name} contains an invalid public-key line"
        )


def write_private_text(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, content.encode())
    finally:
        os.close(descriptor)
    path.chmod(0o600)


def is_runpod(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return bool(source.get("RUNPOD_POD_ID") or source.get("RUNPOD_TCP_PORT_22"))
