#!/usr/bin/env bash
set -Eeuo pipefail

minimum_gib="${1:-30}"

if [[ "${GITHUB_ACTIONS:-}" != "true" || "${RUNNER_OS:-}" != "Linux" \
  || "${ImageOS:-}" != "ubuntu24" ]]; then
  echo "refusing to reclaim disk outside a GitHub-hosted Ubuntu 24 runner" >&2
  exit 2
fi
if [[ ! "${minimum_gib}" =~ ^[0-9]+$ ]] || (( minimum_gib < 20 || minimum_gib > 100 )); then
  echo "minimum free space must be an integer from 20 through 100 GiB" >&2
  exit 2
fi

echo "Disk usage before reclaiming preinstalled toolchains:"
df -h /

# These fixed paths contain optional GitHub-hosted-runner toolchains. Image
# publication runs on a fresh VM and needs none of them. Keep the list explicit:
# this script must never operate on a persistent or self-hosted runner.
preinstalled_paths=(
  /opt/ghc
  /opt/hostedtoolcache
  /usr/local/.ghcup
  /usr/local/lib/android
  /usr/local/share/boost
  /usr/share/dotnet
)
for path in "${preinstalled_paths[@]}"; do
  if [[ -e "${path}" ]]; then
    echo "Removing optional runner toolchain: ${path}"
    sudo rm -rf -- "${path}"
  fi
done

sudo apt-get clean
docker system prune --all --force --volumes

available_kib="$(df -Pk / | awk 'NR == 2 {print $4}')"
required_kib=$((minimum_gib * 1024 * 1024))

echo "Disk usage after cleanup:"
df -h /
if (( available_kib < required_kib )); then
  available_gib=$((available_kib / 1024 / 1024))
  echo "only ${available_gib} GiB is free; image publication requires ${minimum_gib} GiB" >&2
  exit 1
fi
