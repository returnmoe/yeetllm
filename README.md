# YeetLLM

YeetLLM is a RunPod-ready inference appliance that keeps one or more Hugging Face
models resident in separate vLLM engines and exposes all base models and static
LoRA adapters through one OpenAI-compatible endpoint.

```text
SSH -L 8000:127.0.0.1:8000
                 │
                 ▼
        127.0.0.1:8000 router
          ├── 127.0.0.1:8100  base A + LoRAs
          ├── 127.0.0.1:8101  base B + LoRAs
          └── 127.0.0.1:8102  base C + LoRAs
```

There is no model swapping or eviction. Each configured base model has one
long-lived vLLM process and stays loaded until that process or the container
stops.

> YeetLLM is currently on the `development` line. Pushes to the `development`
> branch publish test images without creating a GitHub release.

## Images and CUDA compatibility

YeetLLM does not compile vLLM. It derives from the official vLLM 0.27.0 CUDA
13 image, pinned by its immutable manifest digest. CUDA 12.6 is not supported.

| Image tag | Upstream base | CUDA userspace | Intended RunPod filter |
|---|---|---:|---:|
| `development` | `vllm/vllm-openai:v0.27.0` | 13.0.2 | CUDA 13.0 |

The final Docker stage only adds YeetLLM and its SSH/runtime-tools overlay; it
does not run package installation or compilation against the vLLM filesystem.
Its linked overlay layers allow BuildKit exporters with lazy-base support to
reuse the official layers without unpacking the full CUDA filesystem.

The pinned official amd64 vLLM manifest itself contains a 4.64 GB compressed
layer. YeetLLM preserves that upstream layer verbatim. This avoids creating an
additional oversized custom layer, but it cannot repair a RunPod host or
registry path that rejects the official vLLM layer itself.

Expected development tags after the branch workflow runs:

```text
ghcr.io/returnmoe/yeetllm:development
ghcr.io/returnmoe/yeetllm:dev-<commit SHA>
```

The immutable commit tag is pushed and checked first; only then does the
workflow advance the moving `development` tag. It never creates a release.

Filter the RunPod template for CUDA 13.0. The host driver comes from RunPod;
the container supplies the matching CUDA userspace.
YeetLLM logs both the host driver/capability and `torch.version.cuda` at boot.

## Quick start: one model, one GPU

Create persistent configuration at `/workspace/yeetllm/config.yaml`:

```yaml
models:
  - id: qwen
    model: Qwen/Qwen3-0.6B
    gpus: [0]
    tensor_parallel_size: 1
    pipeline_parallel_size: 1
    trust_remote_code: false
    dtype: auto
    quantization: auto
    gpu_memory_utilization: 0.90

ssh:
  enable: auto
  port: 22
```

The complete annotated schema is in [`config.example.yaml`](config.example.yaml).
Validate without loading or downloading a model:

```bash
yeetllm validate
```

The container command is `yeetllm serve`. Startup readiness stays unhealthy
until every configured engine is ready.

## Multiple models

Assign independent engines to disjoint physical GPUs:

```yaml
startup:
  policy: all
  parallelism: 1

models:
  - id: qwen
    model: Qwen/Qwen3-32B
    gpus: [0, 1]
    tensor_parallel_size: 2
    pipeline_parallel_size: 1

  - id: gemma
    model: google/gemma-3-27b-it
    gpus: [2, 3]
    tensor_parallel_size: 2
    pipeline_parallel_size: 1
```

Both engines remain resident. Requests can alternate `qwen`, `gemma`, and back
again without reloading weights. See
[`examples/multiple-models.yaml`](examples/multiple-models.yaml).

GPU overlap is rejected unless the root option `allow_shared_gpus: true` is set.
Sharing only opts into contention; it does not reserve memory or make two
independent engines safe from OOM.

## One model across multiple GPUs

For a four-GPU tensor-parallel engine:

```yaml
models:
  - id: qwen
    model: Qwen/Qwen3-32B
    gpus: [0, 1, 2, 3]
    tensor_parallel_size: 4
    pipeline_parallel_size: 1
```

Each process receives an argv array and an isolated
`CUDA_VISIBLE_DEVICES=0,1,2,3`; model IDs and `extra_args` are never interpolated
into a shell command. On one node vLLM uses its native multiprocessing path.
`tensor_parallel_size × pipeline_parallel_size` must match the allocated GPU
count (or the total count across nodes in an Instant Cluster).

For a concrete two-GPU DeepSeek V4 example, see
[`examples/deepseek-v4-flash-abliterated.yaml`](examples/deepseek-v4-flash-abliterated.yaml).
It targets two 96 GB RTX Pro 6000 Blackwell GPUs and leaves the checkpoint's
mixed FP4/FP8 format on vLLM's metadata-driven `quantization: auto` path.

## LoRA adapters

Static LoRA loading is enabled per base engine:

```yaml
models:
  - id: qwen
    model: Qwen/Qwen3-32B
    gpus: [0, 1]
    tensor_parallel_size: 2

    lora:
      enabled: true
      max_loras: 4
      max_lora_rank: 64
      max_cpu_loras: 8
      fully_sharded_loras: false
      adapters:
        - id: qwen-code
          model: organization/qwen-code-lora
        - id: qwen-rp
          model: organization/qwen-rp-lora
```

Adapters are resolved into the persistent Hugging Face cache before that engine
starts and are passed with vLLM's current JSON `--lora-modules` form. YeetLLM
requires global uniqueness across base and adapter IDs, verifies
`max_cpu_loras >= max(max_loras, number of static adapters)`, and checks cached
adapter rank against `max_lora_rank`. vLLM 0.27.0 accepts ranks
`1, 8, 16, 32, 64, 128, 256, 320, 512`.

The base and its LoRAs can serve concurrent requests according to vLLM's LoRA
scheduler. Upstream runtime load/unload endpoints are blocked and
`VLLM_ALLOW_RUNTIME_LORA_UPDATING` is removed from engine environments.
YeetLLM does not currently expose a runtime mutation API.

## One SSH tunnel and one API URL

RunPod maps internal TCP 22 to a public port. Open exactly one tunnel:

```bash
ssh -N \
  -L 8000:127.0.0.1:8000 \
  -p "$RUNPOD_TCP_PORT_22" \
  root@"$RUNPOD_PUBLIC_IP"
```

The only client base URL is:

```text
http://127.0.0.1:8000/v1
```

Python with the official OpenAI client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-used-by-the-loopback-appliance",
    base_url="http://127.0.0.1:8000/v1",
)

response = client.chat.completions.create(
    model="qwen-code",
    messages=[{"role": "user", "content": "Write a Python iterator."}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Model catalog

`GET /v1/models` is assembled from the canonical YeetLLM registry:

```json
{
  "object": "list",
  "data": [
    {"id": "qwen", "object": "model", "owned_by": "yeetllm", "parent": null},
    {"id": "qwen-code", "object": "model", "owned_by": "yeetllm", "parent": "qwen"},
    {"id": "qwen-rp", "object": "model", "owned_by": "yeetllm", "parent": "qwen"},
    {"id": "gemma", "object": "model", "owned_by": "yeetllm", "parent": null}
  ]
}
```

The router proxies `/v1/*` generically, selecting an engine from the JSON,
form, multipart, or query-string `model` field. It preserves the original model
ID, upstream status, relevant headers, and streams SSE chunks without collecting
the full response. Response-ID affinity supports follow-up Responses API routes.

`GET /health/live` checks the router process. `GET /health/ready` succeeds only
when the supervisor heartbeat is fresh and every required engine is ready.

### Why a small YeetLLM router?

The current [vLLM Production Stack](https://github.com/vllm-project/production-stack)
router supports static backends, multiple models, and aliases, but its primary
scope is cluster-wide/Kubernetes routing, replicas, and scheduling. YeetLLM needs
one-to-one integration with a local process supervisor, a flattened base+LoRA
catalog, exact adapter-ID preservation, Responses API affinity, and readiness
without synthetic generation requests. The intentionally small proxy here owns
only those appliance concerns; vLLM still owns all inference behavior.

## SSH security and key detection

`ssh.enable: auto` does **not** start sshd merely because the image runs on
RunPod. It starts only after a non-empty public-key source is detected and the
selected key passes `ssh-keygen` validation. Precedence is:

1. `YEETLLM_SSH_AUTHORIZED_KEYS`
2. `SSH_PUBLIC_KEY`
3. `/root/.ssh/authorized_keys`
4. `PUBLIC_KEY`

This covers RunPod's current account-key injection, per-Pod key, and established
container conventions. If auto mode finds no key, it logs a diagnostic and
leaves SSH disabled—there is no port-22 listener. `ssh.enable: true` with no
valid key is a startup error. Worker nodes in an Instant Cluster never start
sshd.

At runtime YeetLLM generates fresh Ed25519 and RSA host private keys under
`/run`, then prints each **public** key and SHA256 fingerprint to container logs.
The image contains no host private keys. Root administration is public-key-only:

```text
PermitRootLogin prohibit-password
AuthenticationMethods publickey
PasswordAuthentication no
KbdInteractiveAuthentication no
AllowTcpForwarding local
GatewayPorts no
AllowAgentForwarding no
AllowStreamLocalForwarding no
X11Forwarding no
PermitTunnel no
PermitUserEnvironment no
```

Model engines and the router run as the unprivileged `vllm` account provided by
the official base image. sshd runs as root because root is the intended
administrator for the disposable Pod.

Model repositories are downloaded explicitly with vLLM's `--download-dir` to:

```text
/workspace/yeetllm/cache/huggingface/hub
```

YeetLLM also pins `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, and `VLLM_CACHE_ROOT` to
their `/workspace/yeetllm/cache` locations for every engine. This prevents an
engine from silently filling the image's smaller ephemeral root filesystem.
During startup, upstream vLLM/download output is prefixed with the engine ID and
sent to container stdout. An additional progress record is emitted every 30
seconds with elapsed time, engine PID, backend probe state, downloaded cache
bytes, and the number of incomplete cache files. The same snapshot appears in
`yeetllm status --json`.

## RunPod deployment

1. Choose the YeetLLM image and require CUDA 13.0 in the template's CUDA
   filter. Make the GHCR package public or configure matching private-registry
   credentials in RunPod, and allocate enough container disk for the roughly
   9 GB uncompressed runtime image.
2. Mount persistent storage at `/workspace`. A Pod volume survives stop/start
   but is deleted with the Pod; a network volume persists independently.
3. Either put `config.yaml` at `/workspace/yeetllm/config.yaml`, set
   `YEETLLM_CONFIG` to another path, or set `YEETLLM_CONFIG_URL` to fetch it
   automatically over HTTPS at every container start.
4. Add only `22/tcp` to the template's exposed TCP ports. Do not expose 8000 or
   any 810x engine port.
5. Enable RunPod SSH (`startSsh: true` where that deployment API exposes it) and
   configure an account public key before starting the Pod. YeetLLM does not
   depend on undocumented `startSsh` runtime behavior; it still requires an
   actually injected key.
6. Set `HF_TOKEN` only if a configured repository is gated/private.
7. Keep the image's default entrypoint and command. A RunPod template start
   command overrides Docker `CMD`, so leave it blank unless intentional.

RunPod's full SSH variables are `RUNPOD_PUBLIC_IP` and
`RUNPOD_TCP_PORT_22`. Account keys added after a Pod starts are not guaranteed to
appear until redeploy; inspect container logs for YeetLLM's selected key source.

### RunPod CLI example

For a basic one-GPU deployment, register your SSH key once:

```bash
runpodctl ssh add-key --key-file "$HOME/.ssh/id_ed25519.pub"
```

Then create the Pod with the development image and the repository's prefilled
one-GPU configuration:

```bash
runpodctl pod create --name yeetllm --image ghcr.io/returnmoe/yeetllm:development --gpu-id "NVIDIA GeForce RTX 4090" --gpu-count 1 --cloud-type SECURE --min-cuda-version 13.0 --container-disk-in-gb 30 --volume-in-gb 100 --volume-mount-path /workspace --ports 22/tcp --env '{"YEETLLM_CONFIG_URL":"https://raw.githubusercontent.com/returnmoe/yeetllm/development/config.example.yaml"}' --ssh=true
```

That is the complete create command: it exposes only SSH, downloads the YAML at
startup, and starts sshd only if RunPod actually injects the registered public
key. Substitute another value from `runpodctl gpu list` if the prefilled GPU is
unavailable. Open the Pod's **Logs** pane in the RunPod Console to see the
runtime-generated SSH host public keys and SHA256 fingerprints.

The following more defensive script uses the same `runpodctl pod create`
interface, fetches the YAML from `CONFIG_URL`, requests RunPod-managed SSH
setup, exposes only `22/tcp`, and also passes the same public key through
`SSH_PUBLIC_KEY`. No interactive Pod setup or pre-seeded configuration file is
required. The network volume persists model caches and the validated downloaded
configuration.

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# Required inputs. runpodctl must already be configured with `runpodctl doctor`
# or `runpodctl config --apiKey ...`.
GPU_ID="${GPU_ID:?set GPU_ID to a value from: runpodctl gpu list}"
NETWORK_VOLUME_ID="${NETWORK_VOLUME_ID:?set NETWORK_VOLUME_ID}"
CONFIG_URL="${CONFIG_URL:?set CONFIG_URL to an HTTPS YAML URL}"

# Optional overrides.
GPU_COUNT="${GPU_COUNT:-1}"
IMAGE="${IMAGE:-ghcr.io/returnmoe/yeetllm:development}"
CONFIG_SHA256="${CONFIG_SHA256:-}"
SSH_PUBLIC_KEY_FILE="${SSH_PUBLIC_KEY_FILE:-${HOME}/.ssh/id_ed25519.pub}"
SSH_PRIVATE_KEY_FILE="${SSH_PRIVATE_KEY_FILE:-${SSH_PUBLIC_KEY_FILE%.pub}}"

for required_command in runpodctl jq ssh-keyscan ssh-keygen; do
  command -v "${required_command}" >/dev/null || {
    echo "${required_command} is required" >&2
    exit 1
  }
done

# Never print the Pod's env field: it can contain a signed configuration URL or
# HF_TOKEN. This summary includes only connection and scheduling information.
print_pod_summary() {
  jq '{
    id,
    name,
    image: (.image // .imageName),
    desiredStatus,
    runtimeStatus,
    gpuCount,
    machine,
    ssh
  }'
}

[[ -s "${SSH_PUBLIC_KEY_FILE}" ]] || {
  echo "missing SSH public key: ${SSH_PUBLIC_KEY_FILE}" >&2
  exit 1
}
[[ "${SSH_PRIVATE_KEY_FILE}" != "${SSH_PUBLIC_KEY_FILE}" ]] || {
  echo "set SSH_PRIVATE_KEY_FILE explicitly when the public-key path has no .pub suffix" >&2
  exit 1
}
[[ -s "${SSH_PRIVATE_KEY_FILE}" ]] || {
  echo "missing SSH private key: ${SSH_PRIVATE_KEY_FILE}" >&2
  exit 1
}
ssh-keygen -l -f "${SSH_PUBLIC_KEY_FILE}" >/dev/null || {
  echo "invalid SSH public key: ${SSH_PUBLIC_KEY_FILE}" >&2
  exit 1
}
[[ "${CONFIG_URL}" == https://* ]] || {
  echo "CONFIG_URL must use HTTPS" >&2
  exit 1
}
if [[ -n "${CONFIG_SHA256}" && ! "${CONFIG_SHA256}" =~ ^[[:xdigit:]]{64}$ ]]; then
  echo "CONFIG_SHA256 must be a 64-character hexadecimal digest" >&2
  exit 1
fi

# Register the public key with the RunPod account before Pod creation. Never
# pass the private-key file here.
runpodctl ssh add-key --key-file "${SSH_PUBLIC_KEY_FILE}"
ssh_public_key="$(<"${SSH_PUBLIC_KEY_FILE}")"
pod_env="$(jq -cn \
  --arg ssh_public_key "${ssh_public_key}" \
  --arg config_url "${CONFIG_URL}" \
  --arg config_sha256 "${CONFIG_SHA256}" \
  '{
    SSH_PUBLIC_KEY: $ssh_public_key,
    YEETLLM_SSH_ENABLE: "true",
    YEETLLM_CONFIG_URL: $config_url
  } + if $config_sha256 == "" then {} else {
    YEETLLM_CONFIG_SHA256: $config_sha256
  } end')"

# --ssh=true sets RunPod's startSsh deployment flag. YeetLLM independently
# requires the valid public key above before it starts its own hardened sshd.
pod_json="$(runpodctl pod create \
  --name yeetllm-development \
  --image "${IMAGE}" \
  --gpu-id "${GPU_ID}" \
  --gpu-count "${GPU_COUNT}" \
  --cloud-type SECURE \
  --min-cuda-version 13.0 \
  --container-disk-in-gb 30 \
  --network-volume-id "${NETWORK_VOLUME_ID}" \
  --volume-mount-path /workspace \
  --ports '22/tcp' \
  --env "${pod_env}" \
  --ssh=true \
  --output=json)"

printf '%s\n' "${pod_json}" | print_pod_summary
POD_ID="$(jq -er '.id' <<<"${pod_json}")"

# Poll the released CLI until RunPod publishes the SSH mapping and sshd returns
# host keys. A cold pull of the large upstream vLLM image can take several
# minutes. The Pod is intentionally left running if this local wait is aborted.
deadline=$((SECONDS + 1200))
known_hosts="$(mktemp -t yeetllm-known-hosts.XXXXXX)"
while ((SECONDS < deadline)); do
  pod_json="$(runpodctl pod get "${POD_ID}" --output=json)"
  runtime_status="$(jq -r \
    '(.runtimeStatus // .desiredStatus // .ssh.status // "unknown") | ascii_downcase' \
    <<<"${pod_json}")"
  RUNPOD_PUBLIC_IP="$(jq -r '.ssh.ip // empty' <<<"${pod_json}")"
  RUNPOD_TCP_PORT_22="$(jq -r '.ssh.port // empty' <<<"${pod_json}")"

  if [[ "${runtime_status}" == "stopped" || "${runtime_status}" == "exited" \
    || "${runtime_status}" == "terminated" ]]; then
    printf '%s\n' "${pod_json}" | print_pod_summary >&2
    echo "Pod stopped before SSH became ready" >&2
    rm -f -- "${known_hosts}"
    exit 1
  fi
  if [[ -n "${RUNPOD_PUBLIC_IP}" && -n "${RUNPOD_TCP_PORT_22}" ]] \
    && ssh-keyscan -T 5 -p "${RUNPOD_TCP_PORT_22}" \
      "${RUNPOD_PUBLIC_IP}" >"${known_hosts}" 2>/dev/null \
    && [[ -s "${known_hosts}" ]]; then
    break
  fi
  echo "Waiting for SSH on Pod ${POD_ID} (status: ${runtime_status})..." >&2
  sleep 5
done

if [[ ! -s "${known_hosts}" ]]; then
  echo "SSH did not become ready within 20 minutes; Pod ${POD_ID} is still allocated" >&2
  echo "Inspect it with: runpodctl pod get ${POD_ID}" >&2
  rm -f -- "${known_hosts}"
  exit 1
fi

printf '%s\n' "${pod_json}" | print_pod_summary

console_url="https://console.runpod.io/pods"
echo
echo "Pod console: ${console_url}"
echo "Open Pod ${POD_ID}, select Logs, and inspect the container log for:"
echo "  [sshd] host public key: ..."
echo "  [sshd] host fingerprint: ..."
echo
echo "Fingerprints presented by the SSH endpoint (UNTRUSTED until compared):"
ssh-keygen -E sha256 -lf "${known_hosts}"
echo "Known-hosts file: ${known_hosts}"
echo
echo "After verifying that fingerprint, start the tunnel with:"
printf 'ssh -N -i %q -o UserKnownHostsFile=%q ' \
  "${SSH_PRIVATE_KEY_FILE}" "${known_hosts}"
printf -- '-o StrictHostKeyChecking=yes '
printf -- '-L 8000:127.0.0.1:8000 -p %q root@%q\n' \
  "${RUNPOD_TCP_PORT_22}" "${RUNPOD_PUBLIC_IP}"
```

RunPod currently documents Pod container and system logs as a Console feature;
the current CLI has no supported `runpodctl pod logs` command. Consequently the
script prints the Pods Console URL and created Pod ID instead of depending on an
undocumented log endpoint. In the Console, expand the Pod, choose **Logs**,
select the container log, and save/copy it locally if a log dump is required.
The lines to retain are the generated host public keys and their `SHA256`
fingerprints—never a private key.

To verify the endpoint independently before the first SSH connection, scan its
presented public keys and compare these fingerprints byte-for-byte with the
Console output:

```bash
known_hosts="$(mktemp)"
ssh-keyscan -p "${RUNPOD_TCP_PORT_22}" "${RUNPOD_PUBLIC_IP}" >"${known_hosts}"
ssh-keygen -E sha256 -lf "${known_hosts}"

# Only after the fingerprints match:
ssh -N \
  -i "${SSH_PRIVATE_KEY_FILE}" \
  -o UserKnownHostsFile="${known_hosts}" \
  -o StrictHostKeyChecking=yes \
  -L 8000:127.0.0.1:8000 \
  -p "${RUNPOD_TCP_PORT_22}" \
  root@"${RUNPOD_PUBLIC_IP}"
```

See RunPod's current
[`runpodctl pod` reference](https://docs.runpod.io/runpodctl/reference/runpodctl-pod),
[SSH-key setup](https://docs.runpod.io/pods/configuration/use-ssh), and
[Pod log documentation](https://docs.runpod.io/pods/manage-pods).

Expected listeners on a normal primary Pod:

```text
0.0.0.0:22          sshd (only when a valid key was detected)
127.0.0.1:8000      YeetLLM router
127.0.0.1:8100      vLLM engine
127.0.0.1:8101      vLLM engine, if configured
```

Check with `ss -ltnp`. `EXPOSE` metadata contains port 22 only.

## Persistent Hugging Face and vLLM data

The image sets:

```text
HF_HOME=/workspace/yeetllm/cache/huggingface
HUGGINGFACE_HUB_CACHE=/workspace/yeetllm/cache/huggingface/hub
VLLM_CACHE_ROOT=/workspace/yeetllm/cache/vllm
TRITON_CACHE_DIR=/workspace/yeetllm/cache/vllm/triton
TORCHINDUCTOR_CACHE_DIR=/workspace/yeetllm/cache/vllm/torchinductor
```

Additional persistent locations are `/workspace/yeetllm/models` and
`/workspace/yeetllm/quantized`. Restarting against the same volume reuses cached
weights and compiled kernels. `trust_remote_code` defaults to false and can only
be enabled explicitly per model.

## Quantization

Set `quantization: auto` (the default) for a repository whose metadata lets vLLM
detect its format, or set an explicit value such as the one documented for the
repository. YeetLLM queries the quantization registry in the installed vLLM
version during validation instead of maintaining a stale independent list. This
supports pre-quantized repositories in formats supported by the installed vLLM,
including the applicable AWQ, GPTQ, BitsAndBytes, FP8, GGUF, compressed-tensors,
and newer registered methods.

YeetLLM does not ship an offline quantizer in this release. Use an official
vLLM/LLM Compressor workflow and persist its output under
`/workspace/yeetllm/quantized`, then point `model` at that local directory.

## RunPod Instant Clusters

Use the same image and configuration on every node. A RunPod network volume is
recommended because it is mounted at `/workspace` across the cluster; do not
assume ordinary per-Pod volume disks are shared.

YeetLLM detects and validates the current RunPod cluster contract:

```text
PRIMARY_ADDR == MASTER_ADDR
PRIMARY_PORT == MASTER_PORT
NODE_ADDR
NODE_RANK
NUM_NODES
NUM_TRAINERS
HOST_NODE_ADDR == PRIMARY_ADDR:PRIMARY_PORT
WORLD_SIZE == NUM_NODES * NUM_TRAINERS
```

It also verifies that `NODE_ADDR` belongs to `ens1`, sets
`NCCL_SOCKET_IFNAME=ens1`, `GLOO_SOCKET_IFNAME=ens1`, and
`VLLM_HOST_IP=NODE_ADDR`, and refuses to route distributed traffic over `eth0`.
Set `NCCL_DEBUG=INFO` temporarily for diagnostics.

YeetLLM uses vLLM 0.27.0's supported multi-node native multiprocessing flags.
Rank 0 runs the private coordinator, each head engine, the router, and optional
SSH. Other ranks run only `vllm serve --headless` workers. Rank 0 releases model
startup in configured batches so `startup.parallelism: 1` remains sequential
across all nodes. Every engine gets a unique rendezvous port derived from the
RunPod primary port.

This is a control/barrier service, not an inference transport; tensor/pipeline
traffic remains vLLM and NCCL. Ray is not forced. Native MP preserves explicit
per-node physical GPU assignment, whereas current vLLM documents that
`--device-ids` has no effect with the Ray executor.

Only rank 0 exposes the API or SSH. Do not publish the control, rendezvous, Ray,
8000, or 810x ports. RunPod's private high-bandwidth network must allow the
rendezvous range. See [`examples/instant-cluster.yaml`](examples/instant-cluster.yaml).

Useful diagnostics:

```bash
env | grep -E '^(PRIMARY|MASTER|NODE|NUM_|WORLD_SIZE|HOST_NODE_ADDR)='
ip -br address show dev ens1
ss -ltnp
NCCL_DEBUG=INFO yeetllm serve
```

## Configuration reference

Global fields:

- `server.host` is deliberately restricted to `127.0.0.1`; default port 8000.
- `startup.policy` is currently `all`; `startup.parallelism` controls how many
  expensive model loads begin together.
- `allow_shared_gpus` defaults to false.
- `cluster.mode` is `auto`, `off`, or `runpod`; interface defaults to `ens1`.
- `ssh.enable` is `auto`, true, or false.

Per-model fields map closely to current vLLM arguments:

- `id`, `model`, `revision`, `tokenizer`, `trust_remote_code`
- `gpus`, `tensor_parallel_size`, `pipeline_parallel_size`
- `distributed_executor_backend` (`auto` or `mp`; Ray is rejected because its
  scheduler does not honor this appliance's physical per-engine GPU mapping)
- `dtype`, `quantization`, `max_model_len`
- `gpu_memory_utilization`, `kv_cache_dtype`
- `lora.enabled`, `max_loras`, `max_lora_rank`, `max_cpu_loras`,
  `fully_sharded_loras`, and static `adapters`
- `extra_args`, an argv list for new vLLM options

YeetLLM rejects security/routing/process-owned flags in `extra_args`, including
host, port, TLS, served name, parallel topology, sleep mode, LoRA mutation, and
cluster rendezvous flags. Other entries are appended verbatim to the argv array.

### Configuration from an HTTPS URL

Set `YEETLLM_CONFIG_URL` to boot directly from remotely hosted YAML. YeetLLM
downloads at most 1 MiB with bounded connect/read timeouts, permits only HTTPS
(including every redirect), keeps TLS certificate verification enabled, applies
the same safe YAML and schema validation as a local file, and only then
atomically replaces `YEETLLM_CONFIG`. The default destination remains
`/workspace/yeetllm/config.yaml`.

```text
YEETLLM_CONFIG_URL=https://example.com/yeetllm/config.yaml
YEETLLM_CONFIG_SHA256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

`YEETLLM_CONFIG_SHA256` is optional and pins the downloaded bytes. Signed HTTPS
URLs are supported, but YeetLLM never prints their query strings. Embedded
`https://user:password@...` credentials are rejected. A failed fetch, digest
mismatch, or invalid document aborts startup before SSH, the router, or model
preparation; YeetLLM never silently falls back to a stale local file.

Precedence is: an explicit CLI `--config` path, then `YEETLLM_CONFIG_URL`, then
the local `YEETLLM_CONFIG` path. When the URL is used, `YEETLLM_CONFIG` is its
atomic persistence destination. On Instant Clusters, every node uses the same
URL and the existing configuration-hash check still prevents mismatched engine
startup.

Simple global environment overrides are intentionally limited:

```text
YEETLLM_CONFIG
YEETLLM_CONFIG_URL
YEETLLM_CONFIG_SHA256
YEETLLM_SSH_ENABLE
YEETLLM_SSH_PORT
YEETLLM_SSH_AUTHORIZED_KEYS
HF_TOKEN
```

See [`.env.example`](.env.example).

## CLI

```bash
yeetllm serve
yeetllm validate
yeetllm validate --gpu-count 4       # deterministic CI/off-host validation
yeetllm status
yeetllm status --json
yeetllm --version
```

`validate` performs no model download or load. `status` reads the atomic runtime
registry/state file and prints base/LoRA-to-engine health mappings.

## Engine failure and shutdown behavior

If an engine exits or its local health endpoint fails, the canonical registry is
marked failed immediately, router requests for its IDs receive a structured 503,
and global readiness fails. YeetLLM does not enter an automatic restart loop.
The router and detected-key SSH service remain available for debugging.

`tini` is the image init and registers as a child subreaper. It is PID 1 under
ordinary Docker; subreaper mode preserves zombie reaping when RunPod inserts a
platform wrapper above the image entrypoint. The Python supervisor launches
children in separate process groups, prefixes their stdout/stderr, forwards
shutdown, waits a grace period, then terminates stragglers. No systemd is
present.

The Docker health check verifies the supervisor heartbeat/phase, every required
engine process, router readiness, and sshd process/configuration when SSH was
actually enabled. It never generates tokens.

## Troubleshooting

### CUDA or driver mismatch

Use the RunPod CUDA filter matching the selected image. Compare boot diagnostics,
`nvidia-smi`, and:

```bash
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

Do not casually set `VLLM_ENABLE_CUDA_COMPATIBILITY=1`; NVIDIA forward
compatibility is limited to specific data-center/professional hardware.

### GPU OOM

Reduce `gpu_memory_utilization`, shorten `max_model_len`, choose a supported
quantized repository, or allocate more GPUs. With multiple resident engines,
confirm their GPU sets do not overlap. Sequential startup reduces host RAM and
download pressure but does not reduce steady-state VRAM.

### Invalid TP/PP size

For one node, TP × PP must equal `len(gpus)`. Across a homogeneous Instant
Cluster, it must equal `len(gpus) × NUM_NODES`. YeetLLM rejects the mismatch
before weights load.

### LoRA rank mismatch

Choose an allowed `max_lora_rank` at least as high as every adapter's `r` and
`rank_pattern`. Cached/local adapters are checked before their base engine
starts.

### Gated Hugging Face repository

Accept the repository license, create a read token, and set `HF_TOKEN` in the
RunPod secret environment. YeetLLM never prints it. Confirm files can be written
under `HF_HOME`.

### Remote model code

If vLLM reports that the repository requires custom code, inspect that code and
then opt in with `trust_remote_code: true` for only that model. It is never
enabled automatically.

### NCCL or interface failures

Verify `NODE_ADDR` is on `ens1`, all nodes have the same config and image, the
private rendezvous ports are reachable, and `/workspace` paths are identical.
Use `NCCL_DEBUG=INFO`; slow `NET/Socket` traffic or an `eth0` address indicates
the wrong interface.

### Worker/coordinator startup

Logs include node rank, primary address, world size, interface, backend, and
per-engine state. Initial coordinator discovery uses the configured startup
timeout; after joining, a worker stops its engines after 60 seconds without the
coordinator instead of spinning forever. Config hashes must match on every node.

### SSH key injection

Look for either `authorized key detected from ...` or `SSH remains disabled` in
container logs. RunPod account keys should exist before Pod creation. The log's
host-key fingerprints are the values to verify on first connection.

### Listener audit

```bash
ss -ltnp
```

Only sshd may listen publicly. Router and engine listeners must show
`127.0.0.1`.

## Building the image

Build the thin derivative of the pinned official vLLM image:

```bash
./docker/build.sh --load
```

No vLLM source compilation occurs. The upstream image reference and digest are
in [`Dockerfile`](Dockerfile); the small build definition is in
[`docker-bake.hcl`](docker-bake.hcl). Pushes to `development` publish both the
moving `development` tag and an immutable `dev-<commit>` tag without creating a
release.

Run local checks with:

```bash
python -m pip install -e '.[dev]'
ruff check .
mypy
pytest
shellcheck docker/*.sh tests/docker/*.sh
```

GPU integration tests are designed for appropriately labeled self-hosted
GitHub runners; normal CI covers schema, command construction, routing,
streaming, process control, SSH policy, and image-level configuration.

## Upstream references

- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
- [vLLM LoRA serving](https://docs.vllm.ai/en/stable/features/lora/)
- [vLLM parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [vLLM quantization](https://docs.vllm.ai/en/stable/features/quantization/)
- [vLLM Production Stack](https://github.com/vllm-project/production-stack)
- [RunPod SSH](https://docs.runpod.io/pods/configuration/use-ssh)
- [RunPod exposed ports](https://docs.runpod.io/pods/configuration/expose-ports)
- [RunPod Instant Cluster configuration](https://docs.runpod.io/instant-clusters/configuration)
- [RunPod storage](https://docs.runpod.io/pods/storage/types)

## Scope

YeetLLM does not train or quantize models, train LoRAs, expose public HTTP by
default, provide a UI, implement model swapping, or replace vLLM's distributed
inference transport. It is deliberately a configuration layer, supervisor, and
small routing registry around upstream vLLM.
