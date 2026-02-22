#!/usr/bin/env bash
set -euo pipefail

OVL_DIR="${1:-/home/thomas-ahle/std_ovl}"
MANIFEST="${OVL_SEMANTIC_MANIFEST:-utils/ovl_semantic/manifest.tsv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=utils/formal_toolchain_resolve.sh
source "$SCRIPT_DIR/formal_toolchain_resolve.sh"

CIRCT_VERILOG="${CIRCT_VERILOG:-$(resolve_default_circt_tool "circt-verilog")}"
CIRCT_TOOL_DIR_DEFAULT="$(derive_tool_dir_from_verilog "$CIRCT_VERILOG")"
CIRCT_BMC="${CIRCT_BMC:-$(resolve_default_circt_tool "circt-bmc" "$CIRCT_TOOL_DIR_DEFAULT")}"
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"
CIRCT_BMC_ARGS="${CIRCT_BMC_ARGS:-}"

OUT="${OUT:-}"
OVL_SEMANTIC_TEST_FILTER="${OVL_SEMANTIC_TEST_FILTER:-.*}"
IGNORE_ASSERTS_UNTIL="${IGNORE_ASSERTS_UNTIL:-1}"
RISING_CLOCKS_ONLY="${RISING_CLOCKS_ONLY:-0}"
ALLOW_MULTI_CLOCK="${ALLOW_MULTI_CLOCK:-1}"
BMC_ASSUME_KNOWN_INPUTS="${BMC_ASSUME_KNOWN_INPUTS:-1}"
FAIL_ON_XPASS="${FAIL_ON_XPASS:-1}"

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
if [[ ! -f "$MANIFEST" ]]; then
  echo "OVL semantic manifest not found: $MANIFEST" >&2
  exit 1
fi

set +e
printf '' | grep -Eq "$OVL_SEMANTIC_TEST_FILTER" 2>/dev/null
filter_ec=$?
set -e
if [[ "$filter_ec" == "2" ]]; then
  echo "invalid OVL_SEMANTIC_TEST_FILTER regex: $OVL_SEMANTIC_TEST_FILTER" >&2
  exit 1
fi

if [[ -n "$OUT" ]]; then
  mkdir -p "$(dirname "$OUT")"
  : > "$OUT"
fi

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

total=0
failures=0
skipped=0
xfail=0
xpass=0

run_case_mode() {
  local case_id="$1"
  local mode="$2"
  local wrapper_sv="$3"
  local ovl_module_v="$4"
  local top_module="$5"
  local bound="$6"
  local known_gap="$7"
  local is_fail_result_gap=0
  local is_pass_result_gap=0
  local is_tool_gap=0
  local is_result_gap=0
  case "$known_gap" in
    1|fail)
      is_fail_result_gap=1
      ;;
    pass)
      is_pass_result_gap=1
      ;;
    tool)
      is_tool_gap=1
      ;;
    any)
      is_fail_result_gap=1
      is_pass_result_gap=1
      is_tool_gap=1
      ;;
  esac

  local wrapper_path="$REPO_ROOT/$wrapper_sv"
  local ovl_path="$OVL_DIR/$ovl_module_v"
  local expect_result=""

  if [[ "$mode" == "pass" ]]; then
    expect_result="UNSAT"
  else
    expect_result="SAT"
  fi

  local mlir="$tmpdir/${case_id}_${mode}.mlir"
  local verilog_log="$tmpdir/${case_id}_${mode}.circt-verilog.log"
  local bmc_log="$tmpdir/${case_id}_${mode}.circt-bmc.log"

  local verilog_args=(--no-uvm-auto-include -DOVL_SVA -DOVL_ASSERT_ON -DOVL_GATING_OFF -I "$OVL_DIR")
  if [[ "$mode" == "fail" ]]; then
    verilog_args+=(-DFAIL)
  fi
  if [[ -n "$CIRCT_VERILOG_ARGS" ]]; then
    read -r -a extra_verilog_args <<< "$CIRCT_VERILOG_ARGS"
    verilog_args+=("${extra_verilog_args[@]}")
  fi

  if ! run_limited "$CIRCT_VERILOG" "${verilog_args[@]}" "$wrapper_path" "$ovl_path" >"$mlir" 2>"$verilog_log"; then
    if [[ "$is_tool_gap" == "1" ]]; then
      xfail=$((xfail + 1))
      if [[ -n "$OUT" ]]; then
        printf 'XFAIL(%s): %s [%s]\n' "$mode" "$case_id" "KNOWN_GAP_TOOL_CIRCT_VERILOG_ERROR" >> "$OUT"
      fi
      return 0
    fi
    failures=$((failures + 1))
    if [[ -n "$OUT" ]]; then
      printf 'FAIL(%s): %s [%s]\n' "$mode" "$case_id" "CIRCT_VERILOG_ERROR" >> "$OUT"
    fi
    return 0
  fi

  local bmc_args=(-b "$bound" "--ignore-asserts-until=$IGNORE_ASSERTS_UNTIL" --module "$top_module")
  if [[ -f "$Z3_LIB" ]]; then
    bmc_args+=("--shared-libs=$Z3_LIB")
  fi
  if [[ "$RISING_CLOCKS_ONLY" == "1" ]]; then
    bmc_args+=(--rising-clocks-only)
  fi
  if [[ "$ALLOW_MULTI_CLOCK" == "1" ]]; then
    bmc_args+=(--allow-multi-clock)
  fi
  if [[ "$BMC_ASSUME_KNOWN_INPUTS" == "1" ]]; then
    bmc_args+=(--assume-known-inputs)
  fi
  if [[ -n "$CIRCT_BMC_ARGS" ]]; then
    read -r -a extra_bmc_args <<< "$CIRCT_BMC_ARGS"
    bmc_args+=("${extra_bmc_args[@]}")
  fi

  local bmc_out=""
  if bmc_out="$(run_limited "$CIRCT_BMC" "${bmc_args[@]}" "$mlir" 2>"$bmc_log")"; then
    :
  else
    if [[ "$is_tool_gap" == "1" ]]; then
      xfail=$((xfail + 1))
      if [[ -n "$OUT" ]]; then
        printf 'XFAIL(%s): %s [%s]\n' "$mode" "$case_id" "KNOWN_GAP_TOOL_CIRCT_BMC_ERROR" >> "$OUT"
      fi
      return 0
    fi
    failures=$((failures + 1))
    if [[ -n "$OUT" ]]; then
      printf 'FAIL(%s): %s [%s]\n' "$mode" "$case_id" "CIRCT_BMC_ERROR" >> "$OUT"
    fi
    return 0
  fi

  local got_result=""
  if grep -q 'BMC_RESULT=SAT' <<< "$bmc_out"; then
    got_result="SAT"
  elif grep -q 'BMC_RESULT=UNSAT' <<< "$bmc_out"; then
    got_result="UNSAT"
  fi

  if [[ "$mode" == "fail" && "$is_fail_result_gap" == "1" ]]; then
    is_result_gap=1
  fi
  if [[ "$mode" == "pass" && "$is_pass_result_gap" == "1" ]]; then
    is_result_gap=1
  fi

  if [[ "$got_result" != "$expect_result" ]]; then
    if [[ "$is_result_gap" == "1" ]]; then
      xfail=$((xfail + 1))
      if [[ -n "$OUT" ]]; then
        printf 'XFAIL(%s): %s [%s expected=%s got=%s]\n' "$mode" "$case_id" "KNOWN_GAP" "$expect_result" "${got_result:-NONE}" >> "$OUT"
      fi
      return 0
    fi
    failures=$((failures + 1))
    if [[ -n "$OUT" ]]; then
      printf 'FAIL(%s): %s [%s expected=%s got=%s]\n' "$mode" "$case_id" "UNEXPECTED_RESULT" "$expect_result" "${got_result:-NONE}" >> "$OUT"
    fi
    return 0
  fi

  if [[ "$is_result_gap" == "1" || "$is_tool_gap" == "1" ]]; then
    xpass=$((xpass + 1))
    if [[ -n "$OUT" ]]; then
      printf 'XPASS(%s): %s\n' "$mode" "$case_id" >> "$OUT"
    fi
    return 0
  fi

  if [[ -n "$OUT" ]]; then
    printf 'PASS(%s): %s\n' "$mode" "$case_id" >> "$OUT"
  fi
  return 0
}

while IFS=$'\t' read -r case_id wrapper_sv ovl_module_v top_module bound known_gap _rest; do
  [[ -z "$case_id" || "${case_id:0:1}" == "#" ]] && continue

  if ! printf '%s\n' "$case_id" | grep -Eq "$OVL_SEMANTIC_TEST_FILTER"; then
    continue
  fi

  if [[ -z "$bound" ]]; then
    bound=8
  fi
  if [[ -z "$known_gap" ]]; then
    known_gap=0
  fi

  total=$((total + 2))
  run_case_mode "$case_id" pass "$wrapper_sv" "$ovl_module_v" "$top_module" "$bound" "$known_gap"
  run_case_mode "$case_id" fail "$wrapper_sv" "$ovl_module_v" "$top_module" "$bound" "$known_gap"
done < "$MANIFEST"

echo "ovl semantic BMC summary: ${total} tests, failures=${failures}, xfail=${xfail}, xpass=${xpass}, skipped=${skipped}"
if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
if [[ "$FAIL_ON_XPASS" == "1" && "$xpass" -ne 0 ]]; then
  exit 1
fi
