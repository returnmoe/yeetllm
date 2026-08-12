from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DEFAULT_CONFIG_PATH = Path("/workspace/yeetllm/config.yaml")
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")
BLOCKED_EXTRA_ARGS = {
    "-n",
    "-dp",
    "-pp",
    "-r",
    "-tp",
    "-asc",
    "--api-server-count",
    "--data-parallel-address",
    "--data-parallel-backend",
    "--data-parallel-hybrid-lb",
    "--data-parallel-rank",
    "--data-parallel-rpc-port",
    "--data-parallel-size",
    "--data-parallel-size-local",
    "--data-parallel-start-rank",
    "--device-ids",
    "--distributed-executor-backend",
    "--dtype",
    "--enable-lora",
    "--enable-sleep-mode",
    "--fully-sharded-loras",
    "--headless",
    "--hf-token",
    "--host",
    "--gpu-memory-utilization",
    "--kv-cache-dtype",
    "--lora-modules",
    "--master-addr",
    "--master-port",
    "--max-cpu-loras",
    "--max-lora-rank",
    "--max-loras",
    "--max-model-len",
    "--model",
    "--nnodes",
    "--node-rank",
    "--pipeline-parallel-size",
    "--port",
    "--quantization",
    "--revision",
    "--root-path",
    "--served-model-name",
    "--ssl-ca-certs",
    "--ssl-certfile",
    "--ssl-keyfile",
    "--tensor-parallel-size",
    "--tokenizer",
    "--trust-remote-code",
    "--uds",
    "--api-key",
    "--config",
    "--grpc",
    "--no-enable-lora",
    "--no-fully-sharded-loras",
    "--no-trust-remote-code",
}
BLOCKED_SHORT_ARG_PREFIXES = ("-asc", "-dp", "-n", "-pp", "-r", "-tp")
BLOCKED_LONG_ARG_PREFIXES = (
    "--api-server-",
    "--config.",
    "--config+",
    "--data-parallel-",
    "--device-",
    "--grpc-",
    "--master-",
    "--node-",
    "--pipeline-parallel-",
    "--served-model-",
    "--ssl-",
    "--tensor-parallel-",
)


class UniqueKeySafeLoader(yaml.SafeLoader):  # type: ignore[misc]
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def construct_unique_mapping(
    loader: UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_unique_mapping
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerConfig(StrictModel):
    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class StartupConfig(StrictModel):
    policy: Literal["all"] = "all"
    parallelism: int = Field(default=1, ge=1)


class ClusterConfig(StrictModel):
    mode: Literal["auto", "off", "runpod"] = "auto"
    nccl_socket_ifname: str = "ens1"
    control_port: int | None = Field(default=None, ge=1, le=65535)
    rendezvous_port_base: int | None = Field(default=None, ge=1, le=65535)
    startup_timeout_seconds: float = Field(default=3600.0, gt=0)

    @field_validator("nccl_socket_ifname")
    @classmethod
    def validate_interface(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
            raise ValueError("cluster interface contains unsupported characters")
        return value


class SSHConfig(StrictModel):
    enable: Literal["auto"] | bool = "auto"
    port: int = Field(default=22, ge=1, le=65535)


class LoRAAdapterConfig(StrictModel):
    id: str
    model: str
    revision: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_model_id(value)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("adapter model must not be empty")
        return value


class LoRAConfig(StrictModel):
    enabled: bool = False
    max_loras: int = Field(default=1, ge=1)
    max_lora_rank: Literal[1, 8, 16, 32, 64, 128, 256, 320, 512] = 16
    max_cpu_loras: int = Field(default=1, ge=1)
    fully_sharded_loras: bool = False
    adapters: list[LoRAAdapterConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_capacity(self) -> LoRAConfig:
        if self.adapters and not self.enabled:
            raise ValueError("LoRA adapters require lora.enabled=true")
        required = max(self.max_loras, len(self.adapters))
        if self.max_cpu_loras < required:
            raise ValueError(
                "max_cpu_loras must be >= max(max_loras, number of static adapters)"
            )
        return self


class ModelConfig(StrictModel):
    id: str
    model: str
    revision: str | None = None
    tokenizer: str | None = None
    trust_remote_code: bool = False
    gpus: list[int]
    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    distributed_executor_backend: str = "auto"
    dtype: str = "auto"
    quantization: str = "auto"
    max_model_len: int | None = Field(default=None, ge=1)
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    kv_cache_dtype: str = "auto"
    extra_args: list[str] = Field(default_factory=list)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return validate_model_id(value)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be empty")
        return value

    @field_validator("gpus")
    @classmethod
    def validate_gpu_list(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("at least one GPU must be assigned")
        if any(index < 0 for index in value):
            raise ValueError("GPU indices must be non-negative")
        if len(value) != len(set(value)):
            raise ValueError("GPU indices may not repeat within one model")
        return value

    @field_validator("extra_args")
    @classmethod
    def validate_extra_args(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value or "\x00" in value:
                raise ValueError("extra_args entries must be nonempty and contain no NUL")
            flag = value.split("=", 1)[0]
            normalized = flag.replace("_", "-") if flag.startswith("--") else flag
            # vLLM's flexible parser accepts abbreviated long options as well as
            # dotted config overrides (for example --config.host). Check the
            # root option so neither form can bypass YeetLLM's isolation flags.
            option_root = normalized.split(".", 1)[0].rstrip("+")
            blocked_long = option_root in BLOCKED_EXTRA_ARGS or (
                option_root.startswith("--")
                and any(
                    candidate.startswith(option_root)
                    for candidate in BLOCKED_EXTRA_ARGS
                    if candidate.startswith("--")
                )
            ) or any(normalized.startswith(prefix) for prefix in BLOCKED_LONG_ARG_PREFIXES)
            blocked_short = any(
                option_root.startswith(prefix) for prefix in BLOCKED_SHORT_ARG_PREFIXES
            )
            if blocked_long or blocked_short:
                raise ValueError(
                    f"{flag} is controlled by YeetLLM and cannot be in extra_args"
                )
        return values

    @field_validator("distributed_executor_backend")
    @classmethod
    def validate_executor_backend(cls, value: str) -> str:
        if value not in {"auto", "mp"}:
            raise ValueError(
                "YeetLLM supports distributed_executor_backend=auto or mp; "
                "Ray does not preserve this appliance's explicit per-engine GPU isolation"
            )
        return value


class YeetConfig(StrictModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    startup: StartupConfig = Field(default_factory=StartupConfig)
    allow_shared_gpus: bool = False
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    models: list[ModelConfig]
    ssh: SSHConfig = Field(default_factory=SSHConfig)

    @model_validator(mode="after")
    def validate_global_ids_and_gpus(self) -> YeetConfig:
        if not self.models:
            raise ValueError("at least one model must be configured")

        owners: dict[str, str] = {}
        gpu_owners: dict[int, str] = {}
        for model in self.models:
            register_unique_id(owners, model.id, f"base model {model.id}")
            for adapter in model.lora.adapters:
                register_unique_id(owners, adapter.id, f"LoRA on {model.id}")
            for gpu in model.gpus:
                previous = gpu_owners.get(gpu)
                if previous is not None and not self.allow_shared_gpus:
                    raise ValueError(
                        f"GPU {gpu} is assigned to both {previous} and {model.id}; "
                        "set allow_shared_gpus=true to opt in"
                    )
                gpu_owners[gpu] = model.id
        return self


class RuntimeValidationError(ValueError):
    """A configuration is structurally valid but incompatible with this host."""


def validate_model_id(value: str) -> str:
    if not MODEL_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "model IDs must be 1-128 characters using letters, digits, . _ : / + or -"
        )
    return value


def register_unique_id(owners: dict[str, str], identifier: str, owner: str) -> None:
    previous = owners.get(identifier)
    if previous is not None:
        raise ValueError(f"duplicate model ID {identifier!r}: {previous} and {owner}")
    owners[identifier] = owner


def config_path(env: Mapping[str, str] | None = None) -> Path:
    source = os.environ if env is None else env
    return Path(source.get("YEETLLM_CONFIG", str(DEFAULT_CONFIG_PATH)))


def load_config(path: Path | None = None, env: Mapping[str, str] | None = None) -> YeetConfig:
    source = os.environ if env is None else env
    selected = config_path(source) if path is None else path
    try:
        raw = yaml.load(
            selected.read_text(encoding="utf-8"),
            Loader=UniqueKeySafeLoader,  # noqa: S506 - subclasses yaml.SafeLoader
        )
    except FileNotFoundError as exc:
        raise RuntimeValidationError(f"configuration file not found: {selected}") from exc
    except yaml.YAMLError as exc:
        raise RuntimeValidationError(f"invalid YAML in {selected}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeValidationError("configuration root must be a YAML mapping")

    data = dict(raw)
    ssh = dict(data.get("ssh") or {})
    if "YEETLLM_SSH_ENABLE" in source:
        ssh["enable"] = parse_bool_or_auto(source["YEETLLM_SSH_ENABLE"])
    if "YEETLLM_SSH_PORT" in source:
        try:
            ssh["port"] = int(source["YEETLLM_SSH_PORT"])
        except ValueError as exc:
            raise RuntimeValidationError("YEETLLM_SSH_PORT must be an integer") from exc
    if ssh:
        data["ssh"] = ssh
    try:
        return YeetConfig.model_validate(data)
    except ValidationError as exc:
        raise RuntimeValidationError(str(exc)) from exc


def parse_bool_or_auto(value: str) -> bool | Literal["auto"]:
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeValidationError("SSH enable value must be auto, true, or false")


def discover_gpu_count() -> int:
    try:
        result = subprocess.run(
            ["/usr/bin/nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeValidationError(f"could not discover GPUs with nvidia-smi: {exc}") from exc
    indices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not indices:
        raise RuntimeValidationError("nvidia-smi reported no GPUs")
    return len(indices)


def installed_quantization_methods() -> set[str] | None:
    try:
        from vllm.model_executor.layers.quantization import (  # type: ignore[import-not-found]
            QUANTIZATION_METHODS,
        )
    except ImportError:
        return None
    return set(QUANTIZATION_METHODS)


def validate_runtime(
    config: YeetConfig,
    *,
    gpu_count: int,
    node_count: int = 1,
    quantization_methods: set[str] | None = None,
    rank_resolver: Callable[[LoRAAdapterConfig], int | None] | None = None,
) -> list[str]:
    warnings: list[str] = []
    for model in config.models:
        missing = [gpu for gpu in model.gpus if gpu >= gpu_count]
        if missing:
            raise RuntimeValidationError(
                f"model {model.id} references nonexistent GPU indices {missing}; "
                f"this node exposes {gpu_count} GPUs"
            )
        expected_world = len(model.gpus) * node_count
        actual_world = model.tensor_parallel_size * model.pipeline_parallel_size
        if actual_world != expected_world:
            raise RuntimeValidationError(
                f"model {model.id}: tensor_parallel_size x pipeline_parallel_size "
                f"must equal allocated GPUs ({actual_world} != {expected_world})"
            )
        if (
            model.quantization != "auto"
            and quantization_methods is not None
            and model.quantization not in quantization_methods
        ):
            raise RuntimeValidationError(
                f"model {model.id}: quantization {model.quantization!r} is not "
                "registered by the installed vLLM"
            )
        if rank_resolver is not None:
            for adapter in model.lora.adapters:
                rank = rank_resolver(adapter)
                if rank is None:
                    warnings.append(
                        f"model {model.id}: rank for uncached adapter {adapter.id} "
                        "could not be checked offline"
                    )
                elif rank > model.lora.max_lora_rank:
                    raise RuntimeValidationError(
                        f"adapter {adapter.id} rank {rank} exceeds max_lora_rank "
                        f"{model.lora.max_lora_rank}"
                    )
    if config.allow_shared_gpus:
        warnings.append(
            "allow_shared_gpus=true: independent vLLM engines may contend or run out of memory"
        )
    return warnings


def local_adapter_rank(adapter: LoRAAdapterConfig) -> int | None:
    path = Path(adapter.model)
    if not path.is_dir():
        return None
    config_file = path / "adapter_config.json"
    if not config_file.is_file():
        return None
    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeValidationError(f"invalid {config_file}: {exc}") from exc
    ranks: list[int] = []
    if isinstance(data.get("r"), int):
        ranks.append(data["r"])
    rank_pattern = data.get("rank_pattern")
    if isinstance(rank_pattern, dict):
        ranks.extend(value for value in rank_pattern.values() if isinstance(value, int))
    return max(ranks) if ranks else None


def normalized_config_hash(config: YeetConfig) -> str:
    import hashlib

    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
