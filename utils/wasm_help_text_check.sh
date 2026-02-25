#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
CHECK_VERILOG="${CHECK_VERILOG:-auto}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-help-text] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi

if [[ "$CHECK_VERILOG" != "auto" && "$CHECK_VERILOG" != "0" && "$CHECK_VERILOG" != "1" ]]; then
  echo "[wasm-help-text] invalid CHECK_VERILOG value: $CHECK_VERILOG (expected auto, 0, or 1)" >&2
  exit 1
fi

bmc_js="$BUILD_DIR/bin/circt-bmc.js"
sim_js="$BUILD_DIR/bin/circt-sim.js"
verilog_js="$BUILD_DIR/bin/circt-verilog.js"

for required in "$bmc_js" "$sim_js"; do
  if [[ ! -f "$required" ]]; then
    echo "[wasm-help-text] missing tool: $required" >&2
    exit 1
  fi
done

if [[ "$CHECK_VERILOG" == "auto" ]]; then
  if [[ -f "$verilog_js" ]]; then
    CHECK_VERILOG=1
  else
    CHECK_VERILOG=0
  fi
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

check_help() {
  local tool_js="$1"
  local token="$2"
  local tag="$3"
  local out="$tmpdir/$tag.help.out"
  local err="$tmpdir/$tag.help.err"

  "$NODE_BIN" "$tool_js" --help >"$out" 2>"$err"
  if [[ ! -s "$out" ]]; then
    echo "[wasm-help-text] $tag --help produced no stdout" >&2
    cat "$err" >&2 || true
    exit 1
  fi
  if ! grep -Fq -- "$token" "$out"; then
    echo "[wasm-help-text] $tag --help missing expected token: $token" >&2
    cat "$out" >&2
    exit 1
  fi
}

echo "[wasm-help-text] validating circt-bmc.js help content"
check_help "$bmc_js" "--emit-smtlib" "circt-bmc"

echo "[wasm-help-text] validating circt-sim.js help content"
check_help "$sim_js" "--mode" "circt-sim"

if [[ "$CHECK_VERILOG" == "1" ]]; then
  if [[ ! -f "$verilog_js" ]]; then
    echo "[wasm-help-text] CHECK_VERILOG=1 but circt-verilog.js is missing: $verilog_js" >&2
    exit 1
  fi
  echo "[wasm-help-text] validating circt-verilog.js help content"
  check_help "$verilog_js" "--ir-llhd" "circt-verilog"
fi

echo "[wasm-help-text] PASS"
