#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
REPO_ROOT="$(checks_repo_root "$SCRIPT_DIR")"
RUNNER="$REPO_ROOT/utils/run_regression_unified.sh"

if [[ ! -x "$RUNNER" ]]; then
  echo "[cvdp-manifest-smoke-check] missing runner: $RUNNER" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cd "$REPO_ROOT"
"$RUNNER" \
  --profile smoke \
  --engine circt \
  --suite-regex '^cvdp_' \
  --dry-run \
  --out-dir "$tmpdir" >/dev/null

plan="$tmpdir/plan.tsv"
if [[ ! -f "$plan" ]]; then
  echo "[cvdp-manifest-smoke-check] missing plan file: $plan" >&2
  exit 1
fi

suite_ids="$(tail -n +2 "$plan" | cut -f1)"

echo "[cvdp-manifest-smoke-check] case: smoke-excludes-full-noncommercial"
if grep -qx 'cvdp_cocotb_noncommercial' <<<"$suite_ids"; then
  echo "[cvdp-manifest-smoke-check] unexpected full lane in smoke profile: cvdp_cocotb_noncommercial" >&2
  cat "$plan" >&2
  exit 1
fi
if grep -qx 'cvdp_golden_smoke' <<<"$suite_ids"; then
  echo "[cvdp-manifest-smoke-check] unexpected full lane in smoke profile: cvdp_golden_smoke" >&2
  cat "$plan" >&2
  exit 1
fi

echo "[cvdp-manifest-smoke-check] case: smoke-includes-smoke25-lanes"
for required in \
  cvdp_cocotb_noncommercial_smoke25 \
  cvdp_cocotb_commercial_smoke25 \
  cvdp_golden_commercial_smoke25 \
  cvdp_golden_noncommercial_smoke25; do
  if ! grep -qx "$required" <<<"$suite_ids"; then
    echo "[cvdp-manifest-smoke-check] missing required smoke lane: $required" >&2
    cat "$plan" >&2
    exit 1
  fi
done

echo "[cvdp-manifest-smoke-check] PASS"
