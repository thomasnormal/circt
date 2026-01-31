#!/usr/bin/env bash
set -euo pipefail

YOSYS_SVA_DIR="${1:-/home/thomas-ahle/yosys/tests/sva}"
CIRCT_VERILOG="${CIRCT_VERILOG:-build/bin/circt-verilog}"
CIRCT_OPT="${CIRCT_OPT:-build/bin/circt-opt}"
CIRCT_LEC="${CIRCT_LEC:-build/bin/circt-lec}"
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"
CIRCT_OPT_ARGS="${CIRCT_OPT_ARGS:-}"
CIRCT_LEC_ARGS="${CIRCT_LEC_ARGS:-}"
DISABLE_UVM_AUTO_INCLUDE="${DISABLE_UVM_AUTO_INCLUDE:-1}"
TEST_FILTER="${TEST_FILTER:-}"
SKIP_VHDL="${SKIP_VHDL:-1}"
LEC_SMOKE_ONLY="${LEC_SMOKE_ONLY:-0}"
LEC_FAIL_ON_INEQ="${LEC_FAIL_ON_INEQ:-1}"
Z3_BIN="${Z3_BIN:-}"
OUT="${OUT:-$PWD/yosys-sva-lec-results.txt}"
KEEP_LOGS_DIR="${KEEP_LOGS_DIR:-}"
LEC_ASSUME_KNOWN_INPUTS="${LEC_ASSUME_KNOWN_INPUTS:-0}"

if [[ ! -d "$YOSYS_SVA_DIR" ]]; then
  echo "yosys SVA directory not found: $YOSYS_SVA_DIR" >&2
  exit 1
fi

if [[ -z "$Z3_BIN" ]]; then
  if command -v z3 >/dev/null 2>&1; then
    Z3_BIN="z3"
  elif [[ -x /home/thomas-ahle/z3-install/bin/z3 ]]; then
    Z3_BIN="/home/thomas-ahle/z3-install/bin/z3"
  elif [[ -x /home/thomas-ahle/z3/build/z3 ]]; then
    Z3_BIN="/home/thomas-ahle/z3/build/z3"
  else
    Z3_BIN="z3"
  fi
fi

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

results_tmp="$tmpdir/results.txt"
touch "$results_tmp"

pass=0
fail=0
error=0
skip=0
total=0

save_logs() {
  if [[ -z "$KEEP_LOGS_DIR" ]]; then
    return
  fi
  mkdir -p "$KEEP_LOGS_DIR"
  cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
  cp -f "$opt_mlir" "$KEEP_LOGS_DIR/${log_tag}.opt.mlir" 2>/dev/null || true
  cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
    2>/dev/null || true
  cp -f "$opt_log" "$KEEP_LOGS_DIR/${log_tag}.circt-opt.log" 2>/dev/null || true
  cp -f "$lec_log" "$KEEP_LOGS_DIR/${log_tag}.circt-lec.log" 2>/dev/null || true
}

for sv in "$YOSYS_SVA_DIR"/*.sv; do
  if [[ ! -f "$sv" ]]; then
    continue
  fi
  base="$(basename "$sv" .sv)"
  if [[ -n "$TEST_FILTER" ]] && ! [[ "$base" =~ $TEST_FILTER ]]; then
    continue
  fi
  log_tag="$base"
  rel_path="${sv#"$YOSYS_SVA_DIR/"}"
  if [[ "$rel_path" != "$sv" ]]; then
    log_tag="${rel_path%.sv}"
  fi
  log_tag="${log_tag//\//__}"
  if [[ "$SKIP_VHDL" == "1" && -f "$YOSYS_SVA_DIR/$base.vhd" ]]; then
    echo "SKIP(vhdl): $base"
    skip=$((skip + 1))
    continue
  fi

  total=$((total + 1))

  mlir="$tmpdir/${base}.mlir"
  opt_mlir="$tmpdir/${base}.opt.mlir"
  verilog_log="$tmpdir/${base}.circt-verilog.log"
  opt_log="$tmpdir/${base}.circt-opt.log"
  lec_log="$tmpdir/${base}.circt-lec.log"

  verilog_args=()
  if [[ "$DISABLE_UVM_AUTO_INCLUDE" == "1" ]]; then
    verilog_args+=("--no-uvm-auto-include")
  fi
  if [[ -n "$CIRCT_VERILOG_ARGS" ]]; then
    read -r -a extra_args <<<"$CIRCT_VERILOG_ARGS"
    verilog_args+=("${extra_args[@]}")
  fi

  if ! "$CIRCT_VERILOG" --ir-llhd "${verilog_args[@]}" "$sv" > "$mlir" \
      2> "$verilog_log"; then
    printf "ERROR\t%s\t%s\n" "$base" "$sv" >> "$results_tmp"
    error=$((error + 1))
    save_logs
    continue
  fi

  opt_args=("--strip-llhd-processes" "--lower-ltl-to-core"
    "--lower-clocked-assert-like")
  if [[ -n "$CIRCT_OPT_ARGS" ]]; then
    read -r -a extra_opt_args <<<"$CIRCT_OPT_ARGS"
    opt_args+=("${extra_opt_args[@]}")
  fi
  opt_args+=("$mlir")

  if ! "$CIRCT_OPT" "${opt_args[@]}" > "$opt_mlir" 2> "$opt_log"; then
    printf "ERROR\t%s\t%s\n" "$base" "$sv" >> "$results_tmp"
    error=$((error + 1))
    save_logs
    continue
  fi

  lec_args=()
  if [[ "$LEC_SMOKE_ONLY" == "1" ]]; then
    lec_args+=("--emit-mlir")
  else
    lec_args+=("--run-smtlib" "--z3-path=$Z3_BIN")
  fi
  if [[ "$LEC_ASSUME_KNOWN_INPUTS" == "1" ]]; then
    lec_args+=("--assume-known-inputs")
  fi
  if [[ -n "$CIRCT_LEC_ARGS" ]]; then
    read -r -a extra_lec_args <<<"$CIRCT_LEC_ARGS"
    lec_args+=("${extra_lec_args[@]}")
  fi
  lec_args+=("-c1=top" "-c2=top" "$opt_mlir" "$opt_mlir")

  lec_out=""
  if lec_out="$($CIRCT_LEC "${lec_args[@]}" 2> "$lec_log")"; then
    lec_status=0
  else
    lec_status=$?
  fi

    if [[ "$LEC_SMOKE_ONLY" == "1" ]]; then
      if [[ "$lec_status" -eq 0 ]]; then
        result="PASS"
      else
        result="ERROR"
      fi
    else
      if grep -q "LEC_RESULT=EQ" <<<"$lec_out"; then
        result="PASS"
      elif grep -q "LEC_RESULT=NEQ" <<<"$lec_out"; then
        result="FAIL"
      else
        result="ERROR"
      fi
    fi

  case "$result" in
    PASS) pass=$((pass + 1)) ;;
    FAIL) fail=$((fail + 1)) ;;
    *) error=$((error + 1)) ;;
  esac

  printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
  echo "$result: $base"
  save_logs
done

sort "$results_tmp" > "$OUT"

echo "yosys LEC summary: total=$total pass=$pass fail=$fail error=$error skip=$skip"
echo "results: $OUT"
