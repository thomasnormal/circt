#!/usr/bin/env bash
# Stable entrypoints for MCY mutation runner policies.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: run_mutation_mcy_examples_api.sh <profile> [options]

Profiles:
  default       Forward options to run_mutation_mcy_examples.sh unchanged.
  native-real   Enforce native backend + real harnesses (no yosys/smoke).

This script provides a stable wrapper contract for automation and other
entrypoint scripts while policy defaults evolve.
USAGE
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

profile="$1"
shift

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_runner="${script_dir}/run_mutation_mcy_examples.sh"

case "$profile" in
  default)
    exec "$base_runner" "$@"
    ;;
  native-real)
    for arg in "$@"; do
      case "$arg" in
        --smoke|--yosys|--mutations-backend=*|--mutations-backend|--native-tests-mode=*|--native-tests-mode)
          echo "native-real profile rejects conflicting option: $arg" >&2
          exit 2
          ;;
      esac
    done
    exec "$base_runner" \
      --mutations-backend native \
      --require-native-backend \
      --native-tests-mode real \
      --native-real-tests-strict \
      --fail-on-native-noop-fallback \
      "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "unknown profile: $profile" >&2
    usage >&2
    exit 2
    ;;
esac
