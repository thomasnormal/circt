#!/usr/bin/env bash
set -euo pipefail

SV_TESTS_DIR="${1:-/home/thomas-ahle/sv-tests}"
TAG_REGEX="${TAG_REGEX:-(^| )16\\.|(^| )9\\.4\\.4}"

# Memory limit settings to prevent system hangs
CIRCT_MEMORY_LIMIT_GB="${CIRCT_MEMORY_LIMIT_GB:-20}"
CIRCT_TIMEOUT_SECS="${CIRCT_TIMEOUT_SECS:-300}"
CIRCT_MEMORY_LIMIT_KB=$((CIRCT_MEMORY_LIMIT_GB * 1024 * 1024))

# Run a command with memory limit
run_limited() {
  (
    ulimit -v $CIRCT_MEMORY_LIMIT_KB 2>/dev/null || true
    timeout --signal=KILL $CIRCT_TIMEOUT_SECS "$@"
  )
}
TEST_FILTER="${TEST_FILTER:-}"
CIRCT_VERILOG="${CIRCT_VERILOG:-build/bin/circt-verilog}"
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"
CIRCT_OPT="${CIRCT_OPT:-build/bin/circt-opt}"
CIRCT_LEC="${CIRCT_LEC:-build/bin/circt-lec}"
CIRCT_OPT_ARGS="${CIRCT_OPT_ARGS:-}"
CIRCT_LEC_ARGS="${CIRCT_LEC_ARGS:-}"
DISABLE_UVM_AUTO_INCLUDE="${DISABLE_UVM_AUTO_INCLUDE:-1}"
UVM_PATH="${UVM_PATH:-$(pwd)/lib/Runtime/uvm}"
UVM_TAG_REGEX="${UVM_TAG_REGEX:-(^| )uvm( |$)}"
INCLUDE_UVM_TAGS="${INCLUDE_UVM_TAGS:-0}"
TAG_REGEX_EFFECTIVE="$TAG_REGEX"
if [[ "$INCLUDE_UVM_TAGS" == "1" ]]; then
  TAG_REGEX_EFFECTIVE="($TAG_REGEX_EFFECTIVE)|$UVM_TAG_REGEX"
fi
LEC_SMOKE_ONLY="${LEC_SMOKE_ONLY:-0}"
# LEC_FAIL_ON_INEQ removed - circt-lec no longer has --fail-on-inequivalent flag
FORCE_LEC="${FORCE_LEC:-0}"
SKIP_SHOULD_FAIL="${SKIP_SHOULD_FAIL:-1}"
Z3_BIN="${Z3_BIN:-}"
OUT="${OUT:-$PWD/sv-tests-lec-results.txt}"
mkdir -p "$(dirname "$OUT")" 2>/dev/null || true
KEEP_LOGS_DIR="${KEEP_LOGS_DIR:-}"
LEC_ASSUME_KNOWN_INPUTS="${LEC_ASSUME_KNOWN_INPUTS:-0}"

if [[ ! -d "$SV_TESTS_DIR/tests" ]]; then
  echo "sv-tests directory not found: $SV_TESTS_DIR" >&2
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

read_meta() {
  local key="$1"
  local file="$2"
  sed -n "s/^[[:space:]]*:${key}:[[:space:]]*//p" "$file" | head -n 1
}

normalize_paths() {
  local root="$1"
  shift
  local out=()
  for item in "$@"; do
    if [[ -z "$item" ]]; then
      continue
    fi
    if [[ "$item" = /* ]]; then
      out+=("$item")
    else
      out+=("$root/$item")
    fi
  done
  printf '%s\n' "${out[@]}"
}

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

while IFS= read -r -d '' sv; do
  tags="$(read_meta tags "$sv")"
  if [[ -z "$tags" ]]; then
    skip=$((skip + 1))
    continue
  fi
  if ! [[ "$tags" =~ $TAG_REGEX_EFFECTIVE ]]; then
    skip=$((skip + 1))
    continue
  fi

  base="$(basename "$sv" .sv)"
  if [[ -n "$TEST_FILTER" ]] && ! [[ "$base" =~ $TEST_FILTER ]]; then
    skip=$((skip + 1))
    continue
  fi

  type="$(read_meta type "$sv")"
  run_lec=1
  if [[ "$type" =~ [Pp]arsing ]]; then
    run_lec=0
    if [[ "$FORCE_LEC" == "1" ]]; then
      run_lec=1
    fi
  fi

  should_fail="$(read_meta should_fail "$sv")"
  should_fail_because="$(read_meta should_fail_because "$sv")"
  if [[ -n "$should_fail_because" ]]; then
    should_fail="1"
  fi
  if [[ "$SKIP_SHOULD_FAIL" == "1" && "$should_fail" == "1" ]]; then
    skip=$((skip + 1))
    continue
  fi

  files_line="$(read_meta files "$sv")"
  incdirs_line="$(read_meta incdirs "$sv")"
  defines_line="$(read_meta defines "$sv")"
  top_module="$(read_meta top_module "$sv")"
  if [[ -z "$top_module" ]]; then
    top_module="top"
  fi

  test_root="$SV_TESTS_DIR/tests"
  log_tag="$base"
  rel_path="${sv#"$test_root/"}"
  if [[ "$rel_path" != "$sv" ]]; then
    log_tag="${rel_path%.sv}"
  fi
  log_tag="${log_tag//\//__}"
  if [[ -z "$files_line" ]]; then
    files=("$sv")
  else
    mapfile -t files < <(normalize_paths "$test_root" $files_line)
  fi
  if [[ -z "$incdirs_line" ]]; then
    incdirs=()
  else
    mapfile -t incdirs < <(normalize_paths "$test_root" $incdirs_line)
  fi
  incdirs+=("$(dirname "$sv")")

  defines=()
  if [[ -n "$defines_line" ]]; then
    for d in $defines_line; do
      defines+=("$d")
    done
  fi

  use_uvm=0
  if [[ "$tags" =~ $UVM_TAG_REGEX ]]; then
    use_uvm=1
  fi
  if [[ "$use_uvm" == "1" && ! -d "$UVM_PATH" ]]; then
    printf "ERROR\t%s\t%s (UVM path not found: %s)\n" \
      "$base" "$sv" "$UVM_PATH" >> "$results_tmp"
    error=$((error + 1))
    total=$((total + 1))
    continue
  fi

  total=$((total + 1))

  mlir="$tmpdir/${base}.mlir"
  opt_mlir="$tmpdir/${base}.opt.mlir"
  verilog_log="$tmpdir/${base}.circt-verilog.log"
  opt_log="$tmpdir/${base}.circt-opt.log"
  lec_log="$tmpdir/${base}.circt-lec.log"

  cmd=("$CIRCT_VERILOG" --ir-hw --timescale=1ns/1ns --single-unit \
    -Wno-implicit-conv -Wno-index-oob -Wno-range-oob -Wno-range-width-oob)
  if [[ "$DISABLE_UVM_AUTO_INCLUDE" == "1" && "$use_uvm" == "0" ]]; then
    cmd+=("--no-uvm-auto-include")
  fi
  if [[ "$use_uvm" == "1" ]]; then
    cmd+=("--uvm-path=$UVM_PATH")
  fi
  if [[ -n "$CIRCT_VERILOG_ARGS" ]]; then
    read -r -a extra_verilog_args <<<"$CIRCT_VERILOG_ARGS"
    cmd+=("${extra_verilog_args[@]}")
  fi
  for inc in "${incdirs[@]}"; do
    cmd+=("-I" "$inc")
  done
  for def in "${defines[@]}"; do
    cmd+=("-D" "$def")
  done
  if [[ -n "$top_module" ]]; then
    cmd+=("--top=$top_module")
  fi
  cmd+=("${files[@]}")

  if ! run_limited "${cmd[@]}" > "$mlir" 2> "$verilog_log"; then
    printf "ERROR\t%s\t%s\n" "$base" "$sv" >> "$results_tmp"
    error=$((error + 1))
    save_logs
    continue
  fi

  opt_cmd=("$CIRCT_OPT" --lower-llhd-ref-ports --strip-llhd-processes
    --strip-llhd-interface-signals --lower-ltl-to-core
    --lower-clocked-assert-like)
  if [[ -n "$CIRCT_OPT_ARGS" ]]; then
    read -r -a extra_opt_args <<<"$CIRCT_OPT_ARGS"
    opt_cmd+=("${extra_opt_args[@]}")
  fi
  opt_cmd+=("$mlir")

  if ! run_limited "${opt_cmd[@]}" > "$opt_mlir" 2> "$opt_log"; then
    printf "ERROR\t%s\t%s\n" "$base" "$sv" >> "$results_tmp"
    error=$((error + 1))
    save_logs
    continue
  fi

  if [[ "$run_lec" == "0" ]]; then
    printf "PASS\t%s\t%s\n" "$base" "$sv" >> "$results_tmp"
    pass=$((pass + 1))
    save_logs
    continue
  fi

  lec_args=()
  if [[ "$LEC_SMOKE_ONLY" == "1" ]]; then
    lec_args+=("--emit-mlir")
  else
    lec_args+=("--run-smtlib" "--z3-path=$Z3_BIN")
    # Note: --fail-on-inequivalent was removed from circt-lec
    # Result checking is done via output parsing below
  fi
  if [[ "$LEC_ASSUME_KNOWN_INPUTS" == "1" ]]; then
    lec_args+=("--assume-known-inputs")
  fi
  if [[ -n "$CIRCT_LEC_ARGS" ]]; then
    read -r -a extra_lec_args <<<"$CIRCT_LEC_ARGS"
    lec_args+=("${extra_lec_args[@]}")
  fi
  lec_args+=("-c1=$top_module" "-c2=$top_module" "$opt_mlir" "$opt_mlir")

  lec_out=""
  if lec_out="$(run_limited $CIRCT_LEC "${lec_args[@]}" 2> "$lec_log")"; then
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
  save_logs
done < <(find "$SV_TESTS_DIR/tests" -type f -name "*.sv" -print0)

sort "$results_tmp" > "$OUT"

echo "sv-tests LEC summary: total=$total pass=$pass fail=$fail error=$error skip=$skip"
echo "results: $OUT"
