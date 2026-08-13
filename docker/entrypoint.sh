#!/usr/bin/env bash
set -Eeuo pipefail

umask 022

vllm_uid="$(id -u vllm)"
vllm_gid="$(id -g vllm)"

# These paths are part of YeetLLM's privilege boundary. Do not accept global
# environment overrides that could make root repair an arbitrary directory.
export VLLM_CACHE_ROOT=/workspace/yeetllm/cache/vllm
export TRITON_CACHE_DIR=/workspace/yeetllm/cache/vllm/triton-v0.27.0
export TORCHINDUCTOR_CACHE_DIR=/workspace/yeetllm/cache/vllm/torchinductor-v0.27.0

as_vllm() {
  /usr/bin/setpriv \
    "--reuid=${vllm_uid}" \
    "--regid=${vllm_gid}" \
    --init-groups \
    --no-new-privs \
    --bounding-set=-all \
    --inh-caps=-all \
    --ambient-caps=-all \
    -- "$@"
}

prepare_persistent_directory() {
  local directory="$1"

  # Creating as the service account is the cleanest path on ordinary volumes
  # and on mounts where root is mapped to an anonymous NFS identity.
  if as_vllm /usr/bin/mkdir -p -- "${directory}" 2>/dev/null \
    && as_vllm /usr/bin/chmod 0755 -- "${directory}" 2>/dev/null; then
    return
  fi

  /usr/bin/mkdir -p -- "${directory}"
  if as_vllm /usr/bin/test -w "${directory}"; then
    return
  fi

  # Local Pod volumes normally permit ownership changes.
  if /usr/bin/chown "${vllm_uid}:${vllm_gid}" -- "${directory}" 2>/dev/null \
    && /usr/bin/chmod 0755 -- "${directory}" 2>/dev/null \
    && as_vllm /usr/bin/test -w "${directory}"; then
    return
  fi

  # Some RunPod/network filesystems root-squash chown. Restrict this fallback
  # to YeetLLM's four data directories and prove service-user access afterward.
  if /usr/bin/chmod 0777 -- "${directory}" 2>/dev/null \
    && as_vllm /usr/bin/test -w "${directory}"; then
    echo "[yeetllm] WARNING: ${directory} rejects chown; using mode 0777 for mounted-volume compatibility"
    return
  fi

  echo "[yeetllm] ERROR: ${directory} is not writable by the vllm service user and its mounted filesystem rejects repair" >&2
  /usr/bin/stat -c '[yeetllm] path=%n owner=%u:%g mode=%a filesystem=%m' -- "${directory}" >&2 || true
  exit 1
}

install -d -m 0755 /run/sshd /run/yeetllm
if ! getent passwd sshd >/dev/null; then
  /usr/sbin/useradd --system --no-create-home --home-dir /run/sshd \
    --shell /usr/sbin/nologin sshd
fi

for directory in \
  /workspace/yeetllm/cache \
  /workspace/yeetllm/cache/huggingface \
  /workspace/yeetllm/cache/vllm \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  /workspace/yeetllm/models \
  /workspace/yeetllm/quantized; do
  prepare_persistent_directory "${directory}"
done

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
