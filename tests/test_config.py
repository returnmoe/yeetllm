from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest
import respx

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


@respx.mock
def test_https_config_url_is_validated_and_persisted_atomically(tmp_path: Path) -> None:
    payload = (
        b"models:\n"
        b"  - id: remote\n"
        b"    model: organization/remote\n"
        b"    gpus: [0]\n"
    )
    route = respx.get("https://config.example/yeetllm.yaml").mock(
        return_value=httpx.Response(200, content=payload)
    )
    destination = tmp_path / "nested" / "config.yaml"
    config = load_config(
        env={
            "YEETLLM_CONFIG": str(destination),
            "YEETLLM_CONFIG_URL": "https://config.example/yeetllm.yaml",
            "YEETLLM_CONFIG_SHA256": hashlib.sha256(payload).hexdigest(),
        }
    )

    assert route.called
    assert config.models[0].id == "remote"
    assert destination.read_bytes() == payload
    assert destination.stat().st_mode & 0o777 == 0o600


def test_config_url_requires_https(tmp_path: Path) -> None:
    with pytest.raises(RuntimeValidationError, match="must use HTTPS"):
        load_config(
            env={
                "YEETLLM_CONFIG": str(tmp_path / "config.yaml"),
                "YEETLLM_CONFIG_URL": "http://config.example/yeetllm.yaml",
            }
        )


@respx.mock
def test_config_url_error_does_not_reveal_query_string(tmp_path: Path) -> None:
    query_value = "presigned-sensitive-value"
    respx.get(f"https://config.example/config.yaml?token={query_value}").mock(
        return_value=httpx.Response(403)
    )

    with pytest.raises(RuntimeValidationError) as captured:
        load_config(
            env={
                "YEETLLM_CONFIG": str(tmp_path / "config.yaml"),
                "YEETLLM_CONFIG_URL": (
                    f"https://config.example/config.yaml?token={query_value}"
                ),
            }
        )
    assert "HTTP 403" in str(captured.value)
    assert query_value not in str(captured.value)


@respx.mock
def test_invalid_remote_config_does_not_replace_last_valid_file(tmp_path: Path) -> None:
    destination = tmp_path / "config.yaml"
    destination.write_text("last known valid bytes\n", encoding="utf-8")
    respx.get("https://config.example/invalid.yaml").mock(
        return_value=httpx.Response(200, text="models: [")
    )

    with pytest.raises(RuntimeValidationError, match="invalid YAML"):
        load_config(
            env={
                "YEETLLM_CONFIG": str(destination),
                "YEETLLM_CONFIG_URL": "https://config.example/invalid.yaml",
            }
        )
    assert destination.read_text(encoding="utf-8") == "last known valid bytes\n"


@respx.mock
def test_config_url_rejects_https_to_http_redirect(tmp_path: Path) -> None:
    respx.get("https://config.example/redirect").mock(
        return_value=httpx.Response(302, headers={"location": "http://unsafe.example/config"})
    )

    with pytest.raises(RuntimeValidationError, match="must use HTTPS"):
        load_config(
            env={
                "YEETLLM_CONFIG": str(tmp_path / "config.yaml"),
                "YEETLLM_CONFIG_URL": "https://config.example/redirect",
            }
        )


@respx.mock
def test_config_url_rejects_digest_mismatch_without_persisting(tmp_path: Path) -> None:
    destination = tmp_path / "config.yaml"
    respx.get("https://config.example/config.yaml").mock(
        return_value=httpx.Response(
            200,
            text=(
                "models:\n"
                "  - id: remote\n"
                "    model: organization/remote\n"
                "    gpus: [0]\n"
            ),
        )
    )

    with pytest.raises(RuntimeValidationError, match="SHA256 does not match"):
        load_config(
            env={
                "YEETLLM_CONFIG": str(destination),
                "YEETLLM_CONFIG_URL": "https://config.example/config.yaml",
                "YEETLLM_CONFIG_SHA256": "0" * 64,
            }
        )
    assert not destination.exists()


@respx.mock
def test_explicit_config_path_takes_precedence_over_url(tmp_path: Path) -> None:
    destination = tmp_path / "config.yaml"
    destination.write_text(
        "models:\n"
        "  - id: local\n"
        "    model: organization/local\n"
        "    gpus: [0]\n",
        encoding="utf-8",
    )
    route = respx.get("https://config.example/config.yaml").mock(
        return_value=httpx.Response(500)
    )

    config = load_config(
        destination,
        {
            "YEETLLM_CONFIG_URL": "https://config.example/config.yaml",
            "YEETLLM_CONFIG_SHA256": "0" * 64,
        },
    )

    assert config.models[0].id == "local"
    assert not route.called


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
    "environment, message",
    [
        ({"lowercase": "1"}, "uppercase shell-name syntax"),
        ({"CUDA_VISIBLE_DEVICES": "7"}, "controlled by YeetLLM"),
        ({"HF_TOKEN": "secret"}, "supplied as a secret outside YAML"),
        ({"NCCL_P2P_DISABLE": "bad\x00value"}, "contains a NUL byte"),
    ],
)
def test_model_environment_rejects_unsafe_entries(
    environment: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        YeetConfig.model_validate(
            {
                "models": [
                    {
                        "id": "base",
                        "model": "org/base",
                        "gpus": [0],
                        "environment": environment,
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
        "--download-dir=/tmp/ephemeral-models",
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
