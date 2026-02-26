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

profiles="$(awk -F'\t' '$1=="ibex_uvm_compile" {print $2; exit}' "$manifest")"
if [[ -z "$profiles" ]]; then
  echo "missing ibex_uvm_compile row" >&2
  exit 1
fi

if [[ ",$profiles," == *,smoke,* ]]; then
  echo "ibex_uvm_compile must not be tagged smoke while compile lane is red" >&2
  echo "profiles: $profiles" >&2
  exit 1
fi

echo "PASS: Ibex UVM compile lane excluded from smoke"
