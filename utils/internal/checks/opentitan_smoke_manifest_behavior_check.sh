#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
repo_root="$(checks_repo_root "$script_dir")"
cd "$repo_root"

manifest="docs/unified_regression_manifest.tsv"
if [[ ! -f "$manifest" ]]; then
  echo "missing manifest: $manifest" >&2
  exit 1
fi

e2e_profiles="$(awk -F'\t' '$1=="opentitan_e2e_smoke" {print $2; exit}' "$manifest")"
if [[ -z "$e2e_profiles" ]]; then
  echo "missing opentitan_e2e_smoke row" >&2
  exit 1
fi

# Smoke profile should only include lanes that are currently expected-green.
# OpenTitan has dedicated smoke lanes for sim/verilog; e2e belongs to nightly/full.
if [[ ",$e2e_profiles," == *,smoke,* ]]; then
  echo "opentitan_e2e_smoke must not be tagged with smoke profile" >&2
  echo "profiles: $e2e_profiles" >&2
  exit 1
fi

sim_profiles="$(awk -F'\t' '$1=="opentitan_sim_smoke" {print $2; exit}' "$manifest")"
verilog_profiles="$(awk -F'\t' '$1=="opentitan_verilog_smoke" {print $2; exit}' "$manifest")"
if [[ -z "$sim_profiles" || -z "$verilog_profiles" ]]; then
  echo "missing OpenTitan smoke lanes" >&2
  exit 1
fi
if [[ ",$sim_profiles," != *,smoke,* ]]; then
  echo "opentitan_sim_smoke must be tagged smoke" >&2
  exit 1
fi
if [[ ",$verilog_profiles," != *,smoke,* ]]; then
  echo "opentitan_verilog_smoke must be tagged smoke" >&2
  exit 1
fi

echo "PASS: OpenTitan smoke profile excludes unstable e2e lane"
