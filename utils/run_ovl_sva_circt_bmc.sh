#!/usr/bin/env bash
set -euo pipefail

OVL_DIR="${1:-/home/thomas-ahle/std_ovl}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=utils/formal_toolchain_resolve.sh
source "$SCRIPT_DIR/formal_toolchain_resolve.sh"

CIRCT_VERILOG="${CIRCT_VERILOG:-$(resolve_default_circt_tool "circt-verilog")}"
CIRCT_TOOL_DIR_DEFAULT="$(derive_tool_dir_from_verilog "$CIRCT_VERILOG")"
CIRCT_BMC="${CIRCT_BMC:-$(resolve_default_circt_tool "circt-bmc" "$CIRCT_TOOL_DIR_DEFAULT")}"
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"
CIRCT_BMC_ARGS="${CIRCT_BMC_ARGS:-}"

OUT="${OUT:-}"
OVL_BMC_TEST_FILTER="${OVL_BMC_TEST_FILTER:-}"
OVL_BMC_PROFILES="${OVL_BMC_PROFILES:-known,xprop}"
BOUND="${BOUND:-10}"
IGNORE_ASSERTS_UNTIL="${IGNORE_ASSERTS_UNTIL:-1}"
TOP="${TOP:-}"
BMC_ASSUME_KNOWN_INPUTS="${BMC_ASSUME_KNOWN_INPUTS:-1}"
BMC_SMOKE_ONLY="${BMC_SMOKE_ONLY:-0}"
RISING_CLOCKS_ONLY="${RISING_CLOCKS_ONLY:-0}"
ALLOW_MULTI_CLOCK="${ALLOW_MULTI_CLOCK:-0}"

# Memory guardrails.
CIRCT_MEMORY_LIMIT_GB="${CIRCT_MEMORY_LIMIT_GB:-20}"
CIRCT_TIMEOUT_SECS="${CIRCT_TIMEOUT_SECS:-300}"
CIRCT_MEMORY_LIMIT_KB=$((CIRCT_MEMORY_LIMIT_GB * 1024 * 1024))
run_limited() {
  (
    ulimit -v "$CIRCT_MEMORY_LIMIT_KB" 2>/dev/null || true
    timeout --signal=KILL "$CIRCT_TIMEOUT_SECS" "$@"
  )
}

Z3_LIB="${Z3_LIB:-/home/thomas-ahle/z3-install/lib64/libz3.so}"

if [[ ! -d "$OVL_DIR" ]]; then
  echo "OVL directory not found: $OVL_DIR" >&2
  exit 1
fi

if [[ -z "$OVL_BMC_TEST_FILTER" ]]; then
  OVL_BMC_TEST_FILTER='.*'
fi

set +e
printf '' | grep -Eq "$OVL_BMC_TEST_FILTER" 2>/dev/null
ovl_filter_ec=$?
set -e
if [[ "$ovl_filter_ec" == "2" ]]; then
  echo "invalid OVL_BMC_TEST_FILTER regex: $OVL_BMC_TEST_FILTER" >&2
  exit 1
fi

if [[ -n "$OUT" ]]; then
  mkdir -p "$(dirname "$OUT")"
  : > "$OUT"
fi

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

mapfile -t ovl_files < <(find "$OVL_DIR" -maxdepth 1 -type f -name 'ovl_*.v' | sort)

if [[ "${#ovl_files[@]}" -eq 0 ]]; then
  echo "no ovl_*.v files found under $OVL_DIR" >&2
  exit 1
fi

IFS=',' read -r -a profiles <<< "$OVL_BMC_PROFILES"
if [[ "${#profiles[@]}" -eq 0 ]]; then
  profiles=(known)
fi

total=0
failures=0
skipped=0

for sv in "${ovl_files[@]}"; do
  base="$(basename "$sv" .v)"
  if ! printf '%s\n' "$base" | grep -Eq "$OVL_BMC_TEST_FILTER"; then
    continue
  fi

  top_module="$TOP"
  if [[ -z "$top_module" ]]; then
    top_module="$base"
  fi

  mlir="$tmpdir/${base}.mlir"
  verilog_log="$tmpdir/${base}.circt-verilog.log"

  verilog_args=(--no-uvm-auto-include -DOVL_SVA -I "$OVL_DIR")
  if [[ -n "$CIRCT_VERILOG_ARGS" ]]; then
    read -r -a extra_verilog_args <<< "$CIRCT_VERILOG_ARGS"
    verilog_args+=("${extra_verilog_args[@]}")
  fi

  frontend_ok=1
  if ! run_limited "$CIRCT_VERILOG" "${verilog_args[@]}" "$sv" >"$mlir" 2>"$verilog_log"; then
    frontend_ok=0
  fi

  for raw_profile in "${profiles[@]}"; do
    profile="${raw_profile// /}"
    [[ -z "$profile" ]] && continue
    total=$((total + 1))

    if [[ "$frontend_ok" != "1" ]]; then
      failures=$((failures + 1))
      if [[ -n "$OUT" ]]; then
        printf 'FAIL(%s): %s [%s]\n' "$profile" "$base" "CIRCT_VERILOG_ERROR" >> "$OUT"
      fi
      continue
    fi

    bmc_args=(-b "$BOUND" "--ignore-asserts-until=$IGNORE_ASSERTS_UNTIL" --module "$top_module")
    if [[ "$BMC_SMOKE_ONLY" == "1" ]]; then
      bmc_args+=(--emit-mlir)
    else
      if [[ -f "$Z3_LIB" ]]; then
        bmc_args+=("--shared-libs=$Z3_LIB")
      fi
    fi
    if [[ "$RISING_CLOCKS_ONLY" == "1" ]]; then
      bmc_args+=(--rising-clocks-only)
    fi
    if [[ "$ALLOW_MULTI_CLOCK" == "1" ]]; then
      bmc_args+=(--allow-multi-clock)
    fi

    assume_known="$BMC_ASSUME_KNOWN_INPUTS"
    case "$profile" in
      known) assume_known=1 ;;
      xprop) assume_known=0 ;;
      auto) ;;
      *)
        failures=$((failures + 1))
        if [[ -n "$OUT" ]]; then
          printf 'FAIL(%s): %s [%s]\n' "$profile" "$base" "INVALID_PROFILE" >> "$OUT"
        fi
        continue
        ;;
    esac
    if [[ "$assume_known" == "1" ]]; then
      bmc_args+=(--assume-known-inputs)
    fi

    if [[ -n "$CIRCT_BMC_ARGS" ]]; then
      read -r -a extra_bmc_args <<< "$CIRCT_BMC_ARGS"
      bmc_args+=("${extra_bmc_args[@]}")
    fi

    bmc_log="$tmpdir/${base}.${profile}.circt-bmc.log"
    if run_limited "$CIRCT_BMC" "${bmc_args[@]}" "$mlir" > /dev/null 2>"$bmc_log"; then
      if [[ -n "$OUT" ]]; then
        printf 'PASS(%s): %s\n' "$profile" "$base" >> "$OUT"
      fi
    else
      failures=$((failures + 1))
      if [[ -n "$OUT" ]]; then
        printf 'FAIL(%s): %s [%s]\n' "$profile" "$base" "CIRCT_BMC_ERROR" >> "$OUT"
      fi
    fi
  done

done

echo "ovl BMC summary: ${total} tests, failures=${failures}, skipped=${skipped}"
if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
