from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from yeetllm.cluster import ClusterInfo, cluster_environment
from yeetllm.config import ModelConfig

PERSISTENT_MODEL_DOWNLOAD_DIR = "/workspace/yeetllm/cache/huggingface/hub"


@dataclass(frozen=True)
class EngineLaunch:
    argv: list[str]
    env: dict[str, str]


def build_engine_launch(
    model: ModelConfig,
    *,
    port: int,
    cluster: ClusterInfo,
    model_index: int,
    adapter_paths: dict[str, Path] | None = None,
) -> EngineLaunch:
    argv = [
        "vllm",
        "serve",
        model.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        model.id,
        "--download-dir",
        PERSISTENT_MODEL_DOWNLOAD_DIR,
        "--tensor-parallel-size",
        str(model.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(model.pipeline_parallel_size),
        "--gpu-memory-utilization",
        str(model.gpu_memory_utilization),
    ]
    append_option(argv, "--revision", model.revision)
    append_option(argv, "--tokenizer", model.tokenizer)
    if model.trust_remote_code:
        argv.append("--trust-remote-code")
    if model.distributed_executor_backend != "auto" and not cluster.enabled:
        append_option(argv, "--distributed-executor-backend", model.distributed_executor_backend)
    if model.dtype != "auto":
        append_option(argv, "--dtype", model.dtype)
    if model.quantization != "auto":
        append_option(argv, "--quantization", model.quantization)
    if model.max_model_len is not None:
        append_option(argv, "--max-model-len", model.max_model_len)
    if model.kv_cache_dtype != "auto":
        append_option(argv, "--kv-cache-dtype", model.kv_cache_dtype)

    if model.lora.enabled:
        argv.extend(
            [
                "--enable-lora",
                "--max-loras",
                str(model.lora.max_loras),
                "--max-lora-rank",
                str(model.lora.max_lora_rank),
                "--max-cpu-loras",
                str(model.lora.max_cpu_loras),
            ]
        )
        if model.lora.fully_sharded_loras:
            argv.append("--fully-sharded-loras")
        if model.lora.adapters:
            argv.append("--lora-modules")
            for adapter in model.lora.adapters:
                path = str((adapter_paths or {}).get(adapter.id, Path(adapter.model)))
                argv.append(
                    json.dumps(
                        {"name": adapter.id, "path": path, "base_model_name": model.id},
                        separators=(",", ":"),
                    )
                )

    if cluster.enabled:
        rendezvous_port = cluster.rendezvous_port_base + model_index
        argv.extend(
            [
                "--distributed-executor-backend",
                "mp",
                "--nnodes",
                str(cluster.node_count),
                "--node-rank",
                str(cluster.node_rank),
                "--master-addr",
                cluster.primary_addr,
                "--master-port",
                str(rendezvous_port),
            ]
        )
        if not cluster.primary:
            argv.append("--headless")
    argv.extend(model.extra_args)

    env = dict(os.environ)
    env.update(cluster_environment(cluster))
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in model.gpus)
    env.pop("VLLM_ALLOW_RUNTIME_LORA_UPDATING", None)
    for name in (
        "YEETLLM_CONFIG_URL",
        "YEETLLM_CONFIG_SHA256",
        "YEETLLM_SSH_AUTHORIZED_KEYS",
        "SSH_PUBLIC_KEY",
        "PUBLIC_KEY",
    ):
        env.pop(name, None)
    env["HF_HOME"] = "/workspace/yeetllm/cache/huggingface"
    env["HUGGINGFACE_HUB_CACHE"] = PERSISTENT_MODEL_DOWNLOAD_DIR
    env["VLLM_CACHE_ROOT"] = "/workspace/yeetllm/cache/vllm"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return EngineLaunch(argv=argv, env=env)


def append_option(argv: list[str], name: str, value: object | None) -> None:
    if value is not None:
        argv.extend([name, str(value)])
