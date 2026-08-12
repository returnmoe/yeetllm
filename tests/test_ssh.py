from __future__ import annotations

from pathlib import Path

import pytest

from yeetllm.config import RuntimeValidationError, SSHConfig
from yeetllm.ssh import prepare_ssh, select_authorized_keys

PUBLIC_ONE = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOne test-one"
PUBLIC_TWO = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITwo test-two"


def test_key_source_precedence(tmp_path: Path) -> None:
    mounted = tmp_path / "authorized_keys"
    mounted.write_text(PUBLIC_TWO, encoding="utf-8")
    selected = select_authorized_keys(
        {
            "YEETLLM_SSH_AUTHORIZED_KEYS": PUBLIC_ONE,
            "SSH_PUBLIC_KEY": PUBLIC_TWO,
            "PUBLIC_KEY": PUBLIC_TWO,
        },
        mounted,
    )
    assert selected == ("YEETLLM_SSH_AUTHORIZED_KEYS", PUBLIC_ONE + "\n")


def test_mounted_key_precedes_public_key(tmp_path: Path) -> None:
    mounted = tmp_path / "authorized_keys"
    mounted.write_text(PUBLIC_ONE, encoding="utf-8")
    selected = select_authorized_keys({"PUBLIC_KEY": PUBLIC_TWO}, mounted)
    assert selected == (str(mounted), PUBLIC_ONE + "\n")


def test_auto_ssh_stays_disabled_without_detected_key(tmp_path: Path) -> None:
    launch = prepare_ssh(
        SSHConfig(enable="auto"),
        primary=True,
        env={},
        runtime_dir=tmp_path / "run",
        sshd_config=tmp_path / "sshd_config",
        mounted_path=tmp_path / "missing",
    )
    assert launch.enabled is False
    assert launch.argv is None


def test_forced_ssh_fails_without_detected_key(tmp_path: Path) -> None:
    with pytest.raises(RuntimeValidationError, match="explicitly enabled"):
        prepare_ssh(
            SSHConfig(enable=True),
            primary=True,
            env={},
            runtime_dir=tmp_path / "run",
            sshd_config=tmp_path / "sshd_config",
            mounted_path=tmp_path / "missing",
        )


def test_auto_ssh_rejects_private_key_material(tmp_path: Path) -> None:
    launch = prepare_ssh(
        SSHConfig(enable="auto"),
        primary=True,
        env={
            "YEETLLM_SSH_AUTHORIZED_KEYS": (
                "-----BEGIN OPENSSH PRIVATE KEY-----\nnot-a-public-key\n"
                "-----END OPENSSH PRIVATE KEY-----"
            )
        },
        runtime_dir=tmp_path / "run",
        mounted_path=tmp_path / "missing",
    )
    assert launch.enabled is False
    assert not (tmp_path / "run" / "authorized_keys").exists()


def test_workers_never_start_ssh() -> None:
    launch = prepare_ssh(SSHConfig(enable=True), primary=False, env={})
    assert launch.enabled is False
