#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
  echo "container static check failed: $*" >&2
  exit 1
}

grep -Eq '^EXPOSE 22$' Dockerfile || fail "Dockerfile must expose SSH"
if grep -Eq '^EXPOSE .*(8000|810[0-9])' Dockerfile; then
  fail "inference HTTP ports must not be exposed"
fi
grep -Fq '"/usr/bin/tini", "-s", "--"' Dockerfile \
  || fail "tini must be the image init and register as a subreaper"
grep -Fq 'VLLM_IMAGE=vllm/vllm-openai:v0.27.0@sha256:' Dockerfile \
  || fail "official vLLM base must be pinned by digest"
grep -Fq 'ai.yeetllm.vllm.provenance="official-vllm-image"' Dockerfile \
  || fail "official vLLM provenance label is missing"
grep -Fq '# syntax=docker/dockerfile:1.18@sha256:' Dockerfile \
  || fail "Dockerfile frontend must be pinned by digest"
if awk '/^FROM \$\{VLLM_IMAGE\}/{final_stage=1; next} final_stage && /^RUN /{exit 1}' Dockerfile; then
  :
else
  fail "the final vLLM stage must remain thin and contain no RUN instruction"
fi
if awk '
  /^FROM \$\{VLLM_IMAGE\}/ { final_stage=1; next }
  final_stage && /^COPY / && $0 !~ /--link/ { exit 1 }
' Dockerfile; then
  :
else
  fail "final-stage copies must use BuildKit linked layers"
fi
if [[ -e docker/prepare-vllm-source.sh || -e docker/vllm-build.patch ]]; then
  fail "vLLM source-build machinery must not be present"
fi
grep -Fq 'host: Literal["127.0.0.1"]' src/yeetllm/config.py \
  || fail "router host default is not structurally loopback-only"
grep -Fq '"--host",' src/yeetllm/commands.py \
  || fail "engine command is missing an explicit host"
grep -Fq '"127.0.0.1",' src/yeetllm/commands.py \
  || fail "engine command is not loopback-only"
grep -Fq "export PYTHONPATH=\"/opt/yeetllm:\${PYTHONPATH}\"" docker/yeetllm \
  || fail "SSH CLI wrapper must establish its own Python module path"

if find . -type f \( -name 'ssh_host_*_key' -o -name 'id_rsa' -o -name 'id_ed25519' \) \
    -print -quit | grep -q .; then
  fail "repository contains an SSH private-key-shaped file"
fi

for directive in \
  'PermitRootLogin prohibit-password' \
  'AuthenticationMethods publickey' \
  'PasswordAuthentication no' \
  'KbdInteractiveAuthentication no' \
  'AllowTcpForwarding local' \
  'GatewayPorts no' \
  'AllowAgentForwarding no' \
  'AllowStreamLocalForwarding no' \
  'X11Forwarding no' \
  'PermitTunnel no' \
  'PermitUserEnvironment no' \
  'SetEnv PYTHONPATH=/opt/yeetllm'; do
  grep -Fqx "${directive}" docker/sshd-yeetllm.conf \
    || fail "missing sshd directive: ${directive}"
done

echo "container static checks passed"
