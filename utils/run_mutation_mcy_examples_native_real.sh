#!/usr/bin/env bash
# Run MCY examples through CIRCT-native mutation generation with real harnesses.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: run_mutation_mcy_examples_native_real.sh [options]

Wrapper around run_mutation_mcy_examples_api.sh native-real.

Defaults:
  --examples-root ~/mcy/examples
  --out-dir /tmp/mcy_examples_native_real_<timestamp>

All extra arguments are forwarded to the native-real API profile.
USAGE
}

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
examples_root_default="${HOME}/mcy/examples"
out_dir_default="/tmp/mcy_examples_native_real_$(date +%s)"

"${script_dir}/run_mutation_mcy_examples_api.sh" native-real \
  --examples-root "${examples_root_default}" \
  --out-dir "${out_dir_default}" \
  "$@"
