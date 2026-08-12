from __future__ import annotations

import os
import pwd
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from yeetllm.config import (
    LoRAAdapterConfig,
    ModelConfig,
    RuntimeValidationError,
    local_adapter_rank,
)
from yeetllm.processes import service_argv


def resolve_model_adapters(model: ModelConfig) -> dict[str, Path]:
    """Resolve static adapters to durable local paths before vLLM starts."""

    resolved: dict[str, Path] = {}
    hf_home = Path(os.environ.get("HF_HOME", "/workspace/yeetllm/cache/huggingface"))
    cache_dir = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")))
    for adapter in model.lora.adapters:
        path = resolve_adapter(adapter, cache_dir)
        rank = local_adapter_rank(adapter.model_copy(update={"model": str(path)}))
        if rank is not None and rank > model.lora.max_lora_rank:
            raise RuntimeValidationError(
                f"adapter {adapter.id} rank {rank} exceeds max_lora_rank "
                f"{model.lora.max_lora_rank}"
            )
        resolved[adapter.id] = path
    return resolved


def resolve_adapter(adapter: LoRAAdapterConfig, cache_dir: Path) -> Path:
    candidate = Path(adapter.model)
    if candidate.is_dir():
        return candidate.resolve()
    if os.geteuid() == 0 and service_account_exists():
        return download_as_service_user(adapter, cache_dir)
    return download_adapter(adapter, cache_dir)


def download_adapter(adapter: LoRAAdapterConfig, cache_dir: Path) -> Path:
    try:
        downloaded = snapshot_download(
            repo_id=adapter.model,
            revision=adapter.revision,
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN") or None,
        )
    except Exception as exc:
        raise RuntimeValidationError(
            f"could not resolve LoRA adapter {adapter.id!r} from {adapter.model!r}: {exc}"
        ) from exc
    return Path(downloaded)


def download_as_service_user(adapter: LoRAAdapterConfig, cache_dir: Path) -> Path:
    argv = [
        sys.executable,
        "-m",
        "yeetllm.adapter_worker",
        "--repo",
        adapter.model,
        "--cache-dir",
        str(cache_dir),
    ]
    if adapter.revision is not None:
        argv.extend(["--revision", adapter.revision])
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": "/home/vllm",
            "USER": "vllm",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
    )
    for name in ("YEETLLM_SSH_AUTHORIZED_KEYS", "SSH_PUBLIC_KEY", "PUBLIC_KEY"):
        environment.pop(name, None)
    result = subprocess.run(  # noqa: S603 - argv-only privilege drop, never a shell
        service_argv(argv),
        env=environment,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()[-2000:]
        raise RuntimeValidationError(
            f"could not resolve LoRA adapter {adapter.id!r} as the service user: {detail}"
        )
    path = Path(result.stdout.strip())
    if not path.is_dir():
        raise RuntimeValidationError(
            f"LoRA adapter worker returned an invalid path for {adapter.id!r}"
        )
    return path


def service_account_exists() -> bool:
    try:
        pwd.getpwnam("vllm")
    except KeyError:
        return False
    return True
