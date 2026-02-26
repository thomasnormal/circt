#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
REPO_ROOT="$(checks_repo_root "$SCRIPT_DIR")"
RUNNER="$REPO_ROOT/utils/run_regression_unified.sh"

if [[ ! -x "$RUNNER" ]]; then
  echo "[sv-tests-sim-smoke-check] missing runner: $RUNNER" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cd "$REPO_ROOT"
"$RUNNER" \
  --profile smoke \
  --engine circt \
  --suite-regex '^sv_tests_sim' \
  --dry-run \
  --out-dir "$tmpdir" >/dev/null

plan="$tmpdir/plan.tsv"
if [[ ! -f "$plan" ]]; then
  echo "[sv-tests-sim-smoke-check] missing plan file: $plan" >&2
  exit 1
fi

suite_ids="$(tail -n +2 "$plan" | cut -f1)"

echo "[sv-tests-sim-smoke-check] case: smoke-uses-dedicated-sv-tests-sim-lane"
if ! grep -qx 'sv_tests_sim_smoke' <<<"$suite_ids"; then
  echo "[sv-tests-sim-smoke-check] missing smoke lane: sv_tests_sim_smoke" >&2
  cat "$plan" >&2
  exit 1
fi
if grep -qx 'sv_tests_sim' <<<"$suite_ids"; then
  echo "[sv-tests-sim-smoke-check] unexpected full lane in smoke profile: sv_tests_sim" >&2
  cat "$plan" >&2
  exit 1
fi

echo "[sv-tests-sim-smoke-check] PASS"
