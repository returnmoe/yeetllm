#!/usr/bin/env bash
set -Eeuo pipefail

umask 022

install -d -m 0755 /run/sshd /run/yeetllm
if ! getent passwd sshd >/dev/null; then
  /usr/sbin/useradd --system --no-create-home --home-dir /run/sshd \
    --shell /usr/sbin/nologin sshd
fi

install -d -m 0755 -o vllm -g root \
  /workspace/yeetllm/cache/huggingface \
  /workspace/yeetllm/cache/vllm \
  /workspace/yeetllm/models \
  /workspace/yeetllm/quantized

echo "[yeetllm] base=vllm/${YEETLLM_VLLM_VERSION:-unknown} expected_cuda=${YEETLLM_EXPECTED_CUDA:-unknown}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[yeetllm] NVIDIA host GPUs:"
  nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader || true
  echo "[yeetllm] NVIDIA host CUDA capability:"
  nvidia-smi | sed -n '1,3p' || true
else
  echo "[yeetllm] WARNING: nvidia-smi is unavailable"
fi
python3 -c 'import torch; print(f"[yeetllm] container torch={torch.__version__} cuda={torch.version.cuda}")'

if [[ $# -eq 0 ]]; then
  set -- yeetllm serve
elif [[ "${1}" == -* || "${1}" == "serve" || "${1}" == "validate" || "${1}" == "status" ]]; then
  set -- yeetllm "$@"
fi

exec "$@"
