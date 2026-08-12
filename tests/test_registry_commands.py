from __future__ import annotations

import json
from pathlib import Path

import pytest

from yeetllm.cluster import ClusterInfo
from yeetllm.commands import build_engine_launch
from yeetllm.config import YeetConfig
from yeetllm.registry import build_registry


def lora_config() -> YeetConfig:
    return YeetConfig.model_validate(
        {
            "models": [
                {
                    "id": "qwen",
                    "model": "Qwen/Qwen3-32B",
                    "gpus": [2, 3],
                    "tensor_parallel_size": 2,
                    "lora": {
                        "enabled": True,
                        "max_loras": 2,
                        "max_lora_rank": 64,
                        "max_cpu_loras": 3,
                        "adapters": [
                            {"id": "qwen-code", "model": "org/code"},
                            {"id": "qwen-rp", "model": "org/rp"},
                        ],
                    },
                },
                {
                    "id": "gemma",
                    "model": "google/gemma",
                    "gpus": [4],
                },
            ]
        }
    )


def test_registry_flattens_base_and_lora_ids() -> None:
    registry = build_registry(lora_config())
    assert list(registry.models) == ["qwen", "qwen-code", "qwen-rp", "gemma"]
    assert registry.models["qwen-code"].engine_id == "qwen"
    assert registry.models["qwen-code"].parent == "qwen"
    assert registry.engines["qwen"].backend_url == "http://127.0.0.1:8100"
    assert registry.engines["gemma"].backend_url == "http://127.0.0.1:8101"


def test_engine_command_is_argv_and_preserves_lora_id() -> None:
    model = lora_config().models[0]
    launch = build_engine_launch(
        model,
        port=8100,
        cluster=ClusterInfo(enabled=False, role="primary"),
        model_index=0,
        adapter_paths={
            "qwen-code": Path("/models/code adapter"),
            "qwen-rp": Path("/models/rp;not-a-shell-command"),
        },
    )
    assert launch.argv[:3] == ["vllm", "serve", "Qwen/Qwen3-32B"]
    assert launch.env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert launch.argv[launch.argv.index("--host") + 1] == "127.0.0.1"
    lora_index = launch.argv.index("--lora-modules")
    modules = [json.loads(value) for value in launch.argv[lora_index + 1 : lora_index + 3]]
    assert modules == [
        {
            "name": "qwen-code",
            "path": "/models/code adapter",
            "base_model_name": "qwen",
        },
        {
            "name": "qwen-rp",
            "path": "/models/rp;not-a-shell-command",
            "base_model_name": "qwen",
        },
    ]
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" not in launch.env


def test_cluster_command_uses_native_mp_and_headless_worker() -> None:
    model = lora_config().models[0].model_copy(
        update={"tensor_parallel_size": 2, "pipeline_parallel_size": 2}
    )
    cluster = ClusterInfo(
        enabled=True,
        role="worker",
        node_rank=1,
        node_count=2,
        trainers_per_node=2,
        world_size=4,
        primary_addr="10.0.0.1",
        node_addr="10.0.0.2",
        rendezvous_port_base=29501,
        interface="ens1",
    )
    launch = build_engine_launch(model, port=8100, cluster=cluster, model_index=3)
    assert "--headless" in launch.argv
    assert launch.argv[launch.argv.index("--distributed-executor-backend") + 1] == "mp"
    assert launch.argv[launch.argv.index("--master-port") + 1] == "29504"
    assert launch.env["NCCL_SOCKET_IFNAME"] == "ens1"
    assert launch.env["VLLM_HOST_IP"] == "10.0.0.2"


@pytest.mark.parametrize("server_port, ssh_port", [(8100, 22), (8000, 8000)])
def test_listener_port_collisions_are_rejected(server_port: int, ssh_port: int) -> None:
    config = YeetConfig.model_validate(
        {
            "server": {"port": server_port},
            "ssh": {"port": ssh_port},
            "models": [{"id": "base", "model": "org/base", "gpus": [0]}],
        }
    )
    with pytest.raises(ValueError, match="assigned to both"):
        build_registry(config)
