from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from yeetllm.config import YeetConfig


@pytest.fixture
def config_factory() -> Callable[..., YeetConfig]:
    def factory(**updates: Any) -> YeetConfig:
        data: dict[str, Any] = {
            "models": [
                {
                    "id": "qwen",
                    "model": "Qwen/Qwen3-0.6B",
                    "gpus": [0],
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                }
            ]
        }
        data.update(updates)
        return YeetConfig.model_validate(data)

    return factory
