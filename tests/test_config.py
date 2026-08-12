from __future__ import annotations

from pathlib import Path

import pytest

from yeetllm.config import RuntimeValidationError, YeetConfig, load_config, validate_runtime


def test_safe_yaml_and_environment_override(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "models:\n"
        "  - id: qwen\n"
        "    model: Qwen/Qwen3-0.6B\n"
        "    gpus: [0]\n"
        "ssh:\n"
        "  enable: false\n",
        encoding="utf-8",
    )
    config = load_config(
        path,
        {"YEETLLM_SSH_ENABLE": "auto", "YEETLLM_SSH_PORT": "2200"},
    )
    assert config.ssh.enable == "auto"
    assert config.ssh.port == 2200
    assert config.models[0].trust_remote_code is False


def test_unsafe_yaml_tag_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("!!python/object/apply:os.system ['id']\n", encoding="utf-8")
    with pytest.raises(RuntimeValidationError, match="invalid YAML"):
        load_config(path, {})


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "models: []\nmodels:\n  - id: qwen\n    model: org/model\n    gpus: [0]\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeValidationError, match="duplicate key"):
        load_config(path, {})


@pytest.mark.parametrize(
    "models, message",
    [
        (
            [
                {"id": "same", "model": "org/a", "gpus": [0]},
                {"id": "same", "model": "org/b", "gpus": [1]},
            ],
            "duplicate model ID",
        ),
        (
            [
                {
                    "id": "base",
                    "model": "org/a",
                    "gpus": [0],
                    "lora": {
                        "enabled": True,
                        "max_loras": 1,
                        "max_cpu_loras": 1,
                        "adapters": [{"id": "base", "model": "org/lora"}],
                    },
                }
            ],
            "duplicate model ID",
        ),
        (
            [
                {"id": "a", "model": "org/a", "gpus": [0]},
                {"id": "b", "model": "org/b", "gpus": [0]},
            ],
            "assigned to both",
        ),
    ],
)
def test_global_validation(models: list[dict[str, object]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        YeetConfig.model_validate({"models": models})


def test_lora_capacity_includes_static_adapters() -> None:
    with pytest.raises(ValueError, match="max_cpu_loras"):
        YeetConfig.model_validate(
            {
                "models": [
                    {
                        "id": "base",
                        "model": "org/base",
                        "gpus": [0],
                        "lora": {
                            "enabled": True,
                            "max_loras": 1,
                            "max_cpu_loras": 1,
                            "adapters": [
                                {"id": "one", "model": "org/one"},
                                {"id": "two", "model": "org/two"},
                            ],
                        },
                    }
                ]
            }
        )


def test_runtime_gpu_and_parallel_validation(config_factory: object) -> None:
    factory = config_factory
    config = factory(  # type: ignore[operator]
        models=[
            {
                "id": "wide",
                "model": "org/wide",
                "gpus": [0, 1],
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
            }
        ]
    )
    with pytest.raises(RuntimeValidationError, match="must equal allocated GPUs"):
        validate_runtime(config, gpu_count=2)

    missing = factory(  # type: ignore[operator]
        models=[{"id": "bad", "model": "org/bad", "gpus": [2]}]
    )
    with pytest.raises(RuntimeValidationError, match="nonexistent GPU"):
        validate_runtime(missing, gpu_count=2)


def test_quantization_uses_installed_registry(config_factory: object) -> None:
    config = config_factory(  # type: ignore[operator]
        models=[
            {
                "id": "quant",
                "model": "org/quant",
                "gpus": [0],
                "quantization": "future-format",
            }
        ]
    )
    with pytest.raises(RuntimeValidationError, match="not registered"):
        validate_runtime(config, gpu_count=1, quantization_methods={"awq", "gptq"})


def test_controlled_extra_argument_is_rejected() -> None:
    with pytest.raises(ValueError, match="controlled by YeetLLM"):
        YeetConfig.model_validate(
            {
                "models": [
                    {
                        "id": "base",
                        "model": "org/base",
                        "gpus": [0],
                        "extra_args": ["--host=0.0.0.0"],
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    "argument",
    [
        "-tp=2",
        "--api-key",
        "--hf-token=secret",
        "--config",
        "--config=/tmp/vllm.yaml",
        "--config.host=0.0.0.0",
        "--conf.host=0.0.0.0",
        "--config+=/tmp/vllm.yaml",
        "--conf=/tmp/vllm.yaml",
        "--grpc",
        "--grpc-port=9000",
        "--gpu_memory_utilization=0.1",
        "--revi=unsafe",
        "--data-parallel-size=2",
        "--data-parallel-future-option=2",
        "-tp2",
        "-dp2",
        "-asc2",
    ],
)
def test_controlled_security_and_topology_aliases_are_rejected(argument: str) -> None:
    with pytest.raises(ValueError, match="controlled by YeetLLM"):
        YeetConfig.model_validate(
            {
                "models": [
                    {
                        "id": "base",
                        "model": "org/base",
                        "gpus": [0],
                        "extra_args": [argument],
                    }
                ]
            }
        )


def test_ray_backend_is_rejected_because_it_cannot_honor_physical_gpu_ids() -> None:
    with pytest.raises(ValueError, match="Ray does not preserve"):
        YeetConfig.model_validate(
            {
                "models": [
                    {
                        "id": "base",
                        "model": "org/base",
                        "gpus": [0],
                        "distributed_executor_backend": "ray",
                    }
                ]
            }
        )
