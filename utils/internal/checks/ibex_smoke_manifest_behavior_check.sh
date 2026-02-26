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

ibex_bmc_profiles="$(awk -F'\t' '$1=="ibex_formal_bmc_smoke" {print $2; exit}' "$manifest")"
ibex_uvm_profiles="$(awk -F'\t' '$1=="ibex_uvm_compile" {print $2; exit}' "$manifest")"

if [[ -z "$ibex_bmc_profiles" || -z "$ibex_uvm_profiles" ]]; then
  echo "missing ibex smoke rows" >&2
  exit 1
fi

if [[ ",$ibex_bmc_profiles," == *,smoke,* ]]; then
  echo "ibex_formal_bmc_smoke must not be tagged with smoke profile" >&2
  echo "profiles: $ibex_bmc_profiles" >&2
  exit 1
fi

if [[ ",$ibex_uvm_profiles," != *,smoke,* ]]; then
  echo "ibex_uvm_compile must remain tagged with smoke profile" >&2
  echo "profiles: $ibex_uvm_profiles" >&2
  exit 1
fi

echo "PASS: Ibex smoke profile excludes unstable formal BMC lane"
