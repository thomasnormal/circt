#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
repo_root="$(checks_repo_root "$script_dir")"
manifest="$repo_root/docs/unified_regression_manifest.tsv"
if [[ ! -f "$manifest" ]]; then
  echo "missing manifest: $manifest" >&2
  exit 1
fi

row="$(awk -F'\t' '$1=="avip_sim_nightly" {print; exit}' "$manifest")"
if [[ -z "$row" ]]; then
  echo "missing avip_sim_nightly row in manifest" >&2
  exit 1
fi

if [[ "$row" != *"SIM_TIMEOUT=600"* ]]; then
  echo "avip_sim_nightly must set SIM_TIMEOUT=600" >&2
  echo "$row" >&2
  exit 1
fi

if [[ "$row" != *"SIM_TIMEOUT_GRACE=120"* ]]; then
  echo "avip_sim_nightly must set SIM_TIMEOUT_GRACE=120" >&2
  echo "$row" >&2
  exit 1
fi

if [[ "$row" != *"MAX_WALL_MS=600000"* ]]; then
  echo "avip_sim_nightly must set MAX_WALL_MS=600000" >&2
  echo "$row" >&2
  exit 1
fi

if [[ "$row" != *"FAIL_ON_ACTIVITY_LIVENESS=1"* ]]; then
  echo "avip_sim_nightly must set FAIL_ON_ACTIVITY_LIVENESS=1" >&2
  echo "$row" >&2
  exit 1
fi

smoke_row="$(awk -F'\t' '$1=="avip_sim_smoke" {print; exit}' "$manifest")"
if [[ -z "$smoke_row" ]]; then
  echo "missing avip_sim_smoke row in manifest" >&2
  exit 1
fi

if [[ "$smoke_row" != *"FAIL_ON_ACTIVITY_LIVENESS=1"* ]]; then
  echo "avip_sim_smoke must set FAIL_ON_ACTIVITY_LIVENESS=1" >&2
  echo "$smoke_row" >&2
  exit 1
fi

echo "PASS: avip_sim timeout+liveness policy is hardened for contended hosts"
