from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from yeetllm.config import YeetConfig


@dataclass(frozen=True)
class ModelRecord:
    id: str
    engine_id: str
    backend_url: str
    kind: str
    parent: str | None


@dataclass(frozen=True)
class EngineRecord:
    id: str
    backend_url: str
    port: int
    model_ids: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class Registry:
    engines: dict[str, EngineRecord]
    models: dict[str, ModelRecord]
    created: int

    def as_state(self) -> dict[str, object]:
        return {
            "created": self.created,
            "engines": {key: asdict(value) for key, value in self.engines.items()},
            "models": {key: asdict(value) for key, value in self.models.items()},
        }

    def openai_models(self) -> dict[str, object]:
        cards: list[dict[str, object]] = []
        for record in self.models.values():
            cards.append(
                {
                    "id": record.id,
                    "object": "model",
                    "created": self.created,
                    "owned_by": "yeetllm",
                    "root": record.id,
                    "parent": record.parent,
                }
            )
        return {"object": "list", "data": cards}


def build_registry(config: YeetConfig, *, first_port: int = 8100) -> Registry:
    engines: dict[str, EngineRecord] = {}
    models: dict[str, ModelRecord] = {}
    for index, model in enumerate(config.models):
        port = first_port + index
        if port > 65535:
            raise ValueError("internal engine port range exceeds 65535")
        backend = f"http://127.0.0.1:{port}"
        selectable = [model.id, *(adapter.id for adapter in model.lora.adapters)]
        engines[model.id] = EngineRecord(model.id, backend, port, tuple(selectable))
        models[model.id] = ModelRecord(model.id, model.id, backend, "base", None)
        for adapter in model.lora.adapters:
            models[adapter.id] = ModelRecord(
                adapter.id, model.id, backend, "lora", model.id
            )
    claimed: dict[int, str] = {
        config.server.port: "router",
        config.ssh.port: "sshd",
    }
    for engine in engines.values():
        previous = claimed.get(engine.port)
        if previous is not None:
            raise ValueError(
                f"port {engine.port} is assigned to both {previous} and engine {engine.id}"
            )
        claimed[engine.port] = f"engine {engine.id}"
    if config.server.port == config.ssh.port:
        raise ValueError(
            f"port {config.server.port} is assigned to both router and sshd"
        )
    return Registry(engines=engines, models=models, created=int(time.time()))
