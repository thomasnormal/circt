#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
repo_root="$(checks_repo_root "$script_dir")"
cd "$repo_root"

SCRIPT="${1:-utils/internal/checks/wasm_configure_behavior_check.sh}"

if [[ ! -x "$SCRIPT" ]]; then
  echo "[wasm-config-behavior-contract] missing executable: $SCRIPT" >&2
  exit 1
fi

required_tokens=(
  'behavior_out="$tmpdir/configure.out"'
  'behavior_err="$tmpdir/configure.err"'
  '"$SCRIPT" >"$behavior_out" 2>"$behavior_err"'
  'grep -Fq -- "missing LLVM source directory" "$behavior_err"'
  'cat "$behavior_err" >&2'
)

for token in "${required_tokens[@]}"; do
  if ! grep -Fq -- "$token" "$SCRIPT"; then
    echo "[wasm-config-behavior-contract] missing token in behavior check script: $token" >&2
    exit 1
  fi
done

if grep -Fq -- "/tmp/wasm-config-behavior" "$SCRIPT"; then
  echo "[wasm-config-behavior-contract] behavior check uses fixed /tmp output paths" >&2
  exit 1
fi

echo "[wasm-config-behavior-contract] PASS"
