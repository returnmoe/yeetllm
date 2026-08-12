from __future__ import annotations

from pathlib import Path

import pytest

from yeetllm.adapters import resolve_model_adapters
from yeetllm.config import RuntimeValidationError, YeetConfig


def adapter_model(path: Path, *, max_rank: int = 64) -> YeetConfig:
    return YeetConfig.model_validate(
        {
            "models": [
                {
                    "id": "base",
                    "model": "org/base",
                    "gpus": [0],
                    "lora": {
                        "enabled": True,
                        "max_loras": 1,
                        "max_lora_rank": max_rank,
                        "max_cpu_loras": 1,
                        "adapters": [{"id": "adapter", "model": str(path)}],
                    },
                }
            ]
        }
    )


def test_local_adapter_is_resolved_and_rank_checked(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        '{"r": 32, "rank_pattern": {"q_proj": 64}}', encoding="utf-8"
    )
    resolved = resolve_model_adapters(adapter_model(adapter).models[0])
    assert resolved == {"adapter": adapter.resolve()}


def test_local_adapter_rank_overflow_fails_before_engine_start(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 64}', encoding="utf-8")
    with pytest.raises(RuntimeValidationError, match="exceeds max_lora_rank"):
        resolve_model_adapters(adapter_model(adapter, max_rank=32).models[0])
