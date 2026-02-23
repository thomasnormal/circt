#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${1:-utils/wasm_runtime_helpers_behavior_check.sh}"

if [[ ! -x "$SCRIPT" ]]; then
  echo "[wasm-runtime-helpers-behavior-contract] missing executable: $SCRIPT" >&2
  exit 1
fi

required_tokens=(
  'BMC_INPUT="${BMC_INPUT:-'
  'SIM_INPUT="${SIM_INPUT:-'
  'if [[ ! -f "$BMC_INPUT" || ! -f "$SIM_INPUT" ]]; then'
  "[wasm-runtime-helpers-behavior] missing baseline test input(s):"
  'case1_out="$tmpdir/case1.out"'
  'case1_err="$tmpdir/case1.err"'
  'case2_out="$tmpdir/case2.out"'
  'case2_err="$tmpdir/case2.err"'
  '"$SCRIPT" >"$case1_out" 2>"$case1_err"'
  '"$SCRIPT" >"$case2_out" 2>"$case2_err"'
  'cat "$case1_err" >&2'
  'grep -Fq -- "[wasm-rg-default] missing test input: $missing_sv" "$case2_err"'
)

for token in "${required_tokens[@]}"; do
  if ! grep -Fq -- "$token" "$SCRIPT"; then
    echo "[wasm-runtime-helpers-behavior-contract] missing token in behavior check script: $token" >&2
    exit 1
  fi
done

if grep -Fq -- "/tmp/wasm-runtime-helper-case" "$SCRIPT"; then
  echo "[wasm-runtime-helpers-behavior-contract] behavior check uses fixed /tmp output paths" >&2
  exit 1
fi

echo "[wasm-runtime-helpers-behavior-contract] PASS"
