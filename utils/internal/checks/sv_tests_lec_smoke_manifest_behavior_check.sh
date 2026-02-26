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

line="$(awk -F'\t' '$1=="sv_tests_lec_smoke" {print $0; exit}' "$manifest")"
if [[ -z "$line" ]]; then
  echo "missing sv_tests_lec_smoke row" >&2
  exit 1
fi

if [[ "$line" != *"TEST_FILTER='^(16\\.2--assert|16\\.2--assume|16\\.2--cover)$'"* ]]; then
  echo "sv_tests_lec_smoke missing curated TEST_FILTER" >&2
  echo "row: $line" >&2
  exit 1
fi

if [[ "$line" != *"LEC_SMOKE_ONLY=1"* ]]; then
  echo "sv_tests_lec_smoke missing LEC_SMOKE_ONLY=1" >&2
  echo "row: $line" >&2
  exit 1
fi

echo "PASS: sv_tests_lec_smoke manifest row uses curated smoke filter"
