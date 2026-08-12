#!/usr/bin/env bash
set -Eeuo pipefail

image="${1:?usage: smoke.sh IMAGE}"

exposed="$(docker image inspect "${image}" --format '{{json .Config.ExposedPorts}}')"
[[ "${exposed}" == '{"22/tcp":{}}' ]] \
  || { echo "unexpected exposed ports: ${exposed}" >&2; exit 1; }

docker run --rm --entrypoint python3 "${image}" -c '
import subprocess

import torch, vllm, yeetllm
from yeetllm.config import SSHConfig, ServerConfig
from yeetllm.processes import service_argv
from yeetllm.router import create_app
from yeetllm.ssh import prepare_ssh
from yeetllm.supervisor import Supervisor

assert ServerConfig().host == "127.0.0.1"
assert callable(create_app)
assert Supervisor is not None
assert torch.version.cuda.startswith(__import__("os").environ["YEETLLM_EXPECTED_CUDA"])
assert not prepare_ssh(SSHConfig(), primary=True, env={}).enabled
identity = subprocess.run(
    service_argv(["/usr/bin/id", "-u"]), check=True, capture_output=True, text=True
)
assert identity.stdout.strip() == "2000"
print(vllm.__version__, yeetllm.__version__, torch.version.cuda)
'

docker run --rm --entrypoint bash "${image}" -c '
set -Eeuo pipefail
if find /etc/ssh -maxdepth 1 -name "ssh_host_*_key" -print -quit | grep -q .; then
  echo "image contains a baked SSH host private key" >&2
  exit 1
fi
install -d -m 0700 /run/yeetllm/ssh
install -d -m 0755 /run/sshd
getent passwd sshd >/dev/null \
  || useradd --system --no-create-home --home-dir /run/sshd --shell /usr/sbin/nologin sshd
touch /run/yeetllm/authorized_keys
chmod 0600 /run/yeetllm/authorized_keys
ssh-keygen -q -t ed25519 -N "" -f /run/yeetllm/ssh/ssh_host_ed25519_key
ssh-keygen -q -t rsa -b 3072 -N "" -f /run/yeetllm/ssh/ssh_host_rsa_key
/usr/sbin/sshd -t -f /etc/ssh/sshd_config
YEETLLM_SSH_AUTHORIZED_KEYS="$(cat /run/yeetllm/ssh/ssh_host_ed25519_key.pub)" \
  python3 -c "from yeetllm.config import SSHConfig; from yeetllm.ssh import prepare_ssh; launch = prepare_ssh(SSHConfig(), primary=True); assert launch.enabled; assert launch.key_source is not None"
if python3 -m yeetllm.healthcheck; then
  echo "healthcheck unexpectedly passed without supervisor state" >&2
  exit 1
fi
'

entrypoint="$(docker image inspect "${image}" --format '{{json .Config.Entrypoint}}')"
[[ "${entrypoint}" == '["/usr/bin/tini","--","/usr/local/bin/yeetllm-entrypoint"]' ]] \
  || { echo "unexpected entrypoint: ${entrypoint}" >&2; exit 1; }

ssh_temp="$(mktemp -d)"
ssh_container=""
cleanup_ssh_test() {
  if [[ -n "${ssh_container}" ]]; then
    docker rm -f "${ssh_container}" >/dev/null 2>&1 || true
  fi
  rm -rf -- "${ssh_temp}"
}
trap cleanup_ssh_test EXIT

ssh-keygen -q -t ed25519 -N "" -f "${ssh_temp}/client_key"
client_public_key="$(<"${ssh_temp}/client_key.pub")"
ssh_container="$(
  docker run -d \
    --publish 127.0.0.1::22 \
    --env "YEETLLM_SSH_AUTHORIZED_KEYS=${client_public_key}" \
    "${image}" \
    python3 -c '
import os
from yeetllm.config import SSHConfig
from yeetllm.ssh import prepare_ssh

launch = prepare_ssh(SSHConfig(), primary=True)
assert launch.enabled and launch.argv
os.execvpe(launch.argv[0], launch.argv, os.environ)
'
)"
ssh_port="$(docker port "${ssh_container}" 22/tcp | awk -F: 'NR == 1 {print $NF}')"
[[ "${ssh_port}" =~ ^[0-9]+$ ]] \
  || { echo "could not resolve the SSH smoke-test port" >&2; exit 1; }

ssh_result=""
for _ in {1..100}; do
  if ssh_result="$(
    ssh \
      -i "${ssh_temp}/client_key" \
      -p "${ssh_port}" \
      -o BatchMode=yes \
      -o ConnectTimeout=1 \
      -o IdentitiesOnly=yes \
      -o LogLevel=ERROR \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      root@127.0.0.1 /usr/bin/id -u 2>/dev/null
  )"; then
    break
  fi
  sleep 0.1
done
[[ "${ssh_result}" == "0" ]] \
  || { echo "public-key-only root SSH smoke test failed" >&2; exit 1; }

docker rm -f "${ssh_container}" >/dev/null
ssh_container=""
cleanup_ssh_test
trap - EXIT

signal_container=""
cleanup_signal_container() {
  if [[ -n "${signal_container}" ]]; then
    docker rm -f "${signal_container}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_signal_container EXIT

signal_container="$(
  docker run -d "${image}" bash -c \
    'trap "exit 42" TERM; echo signal-ready; while :; do sleep 1; done'
)"
signal_ready=false
for _ in {1..100}; do
  if docker logs "${signal_container}" 2>&1 | grep -Fq 'signal-ready'; then
    signal_ready=true
    break
  fi
  sleep 0.1
done
[[ "${signal_ready}" == true ]] \
  || { echo "entrypoint signal-test container did not become ready" >&2; exit 1; }
docker stop --time 5 "${signal_container}" >/dev/null
signal_exit="$(docker inspect "${signal_container}" --format '{{.State.ExitCode}}')"
[[ "${signal_exit}" == 42 ]] \
  || { echo "tini did not forward SIGTERM cleanly: exit=${signal_exit}" >&2; exit 1; }
docker rm "${signal_container}" >/dev/null
signal_container=""
trap - EXIT

echo "image smoke checks passed: ${image}"
