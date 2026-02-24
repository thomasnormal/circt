#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
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

echo "PASS: avip_sim_nightly timeout policy is hardened for contended hosts"
