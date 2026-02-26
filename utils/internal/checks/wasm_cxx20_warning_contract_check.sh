#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
repo_root="$(checks_repo_root "$script_dir")"
cd "$repo_root"

SCRIPT="${1:-utils/wasm_cxx20_warning_check.sh}"

if [[ ! -x "$SCRIPT" ]]; then
  echo "[wasm-cxx20-warn-contract] missing executable: $SCRIPT" >&2
  exit 1
fi

required_tokens=(
  "CMAKE_CXX_STANDARD:STRING="
  "validate_positive_int_env"
  'validate_positive_int_env "NINJA_JOBS" "$NINJA_JOBS"'
  'invalid $name value'
  "cache C++ standard is non-numeric"
  "build C++ standard is below 20"
  "std_flag_is_cpp20_or_newer"
  "is_emscripten_cpp_compiler"
  "failed to query compile commands for circt-tblgen"
  "ambiguous-reversed-operator"
  "c++20-extensions"
  "-std="
  'grep -Eiq -- "(^|[^[:alpha:]])warning:" "$log"'
  "FIRRTLAnnotationsGen.cpp"
  "FIRRTLIntrinsicsGen.cpp"
  "circt-tblgen.cpp"
  'compile command for $src is missing a -std=... flag'
  'compile command for $src does not use C++20-or-newer'
  '[wasm-cxx20-warn] failed to clean rebuild targets'
  '[wasm-cxx20-warn] missing compile command for'
  "does not appear to use Emscripten em++"
)

for token in "${required_tokens[@]}"; do
  if ! grep -Fq -- "$token" "$SCRIPT"; then
    echo "[wasm-cxx20-warn-contract] missing token in warning check script: $token" >&2
    exit 1
  fi
done

echo "[wasm-cxx20-warn-contract] PASS"
