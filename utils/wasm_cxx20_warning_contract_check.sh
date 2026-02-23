#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${1:-utils/wasm_cxx20_warning_check.sh}"

if [[ ! -x "$SCRIPT" ]]; then
  echo "[wasm-cxx20-warn-contract] missing executable: $SCRIPT" >&2
  exit 1
fi

required_tokens=(
  "CMAKE_CXX_STANDARD:STRING=20"
  "ambiguous-reversed-operator"
  "c++20-extensions"
  "-std=c++20"
  'grep -Eiq -- "(^|[^[:alpha:]])warning:" "$log"'
  "FIRRTLAnnotationsGen.cpp"
  "FIRRTLIntrinsicsGen.cpp"
  "circt-tblgen.cpp"
  'grep -F -- "$src" "$cmd_dump" | grep -Fq -- "-std=c++20"'
  '[wasm-cxx20-warn] failed to clean rebuild targets'
)

for token in "${required_tokens[@]}"; do
  if ! grep -Fq -- "$token" "$SCRIPT"; then
    echo "[wasm-cxx20-warn-contract] missing token in warning check script: $token" >&2
    exit 1
  fi
done

echo "[wasm-cxx20-warn-contract] PASS"
