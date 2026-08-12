#!/usr/bin/env bash
set -Eeuo pipefail

image="${1:?usage: run.sh IMAGE CONFIG_FILE}"
config_file="${2:?usage: run.sh IMAGE CONFIG_FILE}"
expected_models="${YEETLLM_EXPECT_MODELS:?set YEETLLM_EXPECT_MODELS to a comma-separated list}"
container="yeetllm-gpu-${RANDOM}"
model_list_pattern='^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}(,[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127})*$'

config_file="$(realpath -e -- "${config_file}")"
[[ -f "${config_file}" && "${config_file}" != *','* ]] \
  || { echo "CONFIG_FILE must resolve to a regular path without commas" >&2; exit 2; }
[[ "${expected_models}" =~ ${model_list_pattern} ]] \
  || { echo "YEETLLM_EXPECT_MODELS contains an invalid model ID" >&2; exit 2; }

cleanup() {
  docker rm -f "${container}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d \
  --name "${container}" \
  --gpus all \
  --pull always \
  --mount "type=bind,src=${config_file},dst=/workspace/yeetllm/config.yaml,readonly" \
  --health-start-period 60m \
  "${image}" >/dev/null

deadline=$((SECONDS + 3600))
until [[ "$(docker inspect "${container}" --format '{{.State.Health.Status}}')" == "healthy" ]]; do
  if (( SECONDS >= deadline )); then
    docker logs "${container}"
    echo "GPU profile did not become healthy" >&2
    exit 1
  fi
  if [[ "$(docker inspect "${container}" --format '{{.State.Status}}')" != "running" ]]; then
    docker logs "${container}"
    echo "container exited during startup" >&2
    exit 1
  fi
  sleep 5
done

initial_pids="$(docker exec "${container}" yeetllm status --json \
  | jq -S '[.engines[] | .pid]')"

IFS=',' read -r -a model_ids <<<"${expected_models}"
catalog="$(docker exec "${container}" curl -fsS http://127.0.0.1:8000/v1/models)"
for model in "${model_ids[@]}"; do
  jq -e --arg model "${model}" '.data | any(.id == $model)' <<<"${catalog}" >/dev/null
done

for _ in 1 2 3; do
  for model in "${model_ids[@]}"; do
    payload="$(
      jq -cn --arg model "${model}" \
        '{model:$model,messages:[{role:"user",content:"Reply with OK"}],max_tokens:4}'
    )"
    docker exec "${container}" curl -fsS http://127.0.0.1:8000/v1/chat/completions \
      -H 'content-type: application/json' \
      --data-binary "${payload}" \
      | jq -e '.choices | length > 0' >/dev/null
  done
done

stream_payload="$(
  jq -cn --arg model "${model_ids[0]}" \
    '{model:$model,messages:[{role:"user",content:"Reply with OK"}],max_tokens:4,stream:true}'
)"
docker exec "${container}" curl -fsSN http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  --data-binary "${stream_payload}" \
  | grep -q '^data:'

final_pids="$(docker exec "${container}" yeetllm status --json \
  | jq -S '[.engines[] | .pid]')"
[[ "${initial_pids}" == "${final_pids}" ]] \
  || { echo "engine PID changed (unexpected reload)" >&2; exit 1; }

public_http="$(docker exec "${container}" ss -ltnH \
  | awk '$4 !~ /^(127\.0\.0\.1|\[::1\]):/ && $4 !~ /:22$/ { print }')"
[[ -z "${public_http}" ]] \
  || { echo "unexpected public listeners:" >&2; echo "${public_http}" >&2; exit 1; }

echo "GPU integration profile passed"
