#!/usr/bin/env bash
set -euo pipefail

VERIF_DIR="${1:-/home/thomas-ahle/verilator-verification}"
shift || true
BOUND="${BOUND:-10}"
IGNORE_ASSERTS_UNTIL="${IGNORE_ASSERTS_UNTIL:-1}"
RISING_CLOCKS_ONLY="${RISING_CLOCKS_ONLY:-0}"
ALLOW_MULTI_CLOCK="${ALLOW_MULTI_CLOCK:-0}"
Z3_LIB="${Z3_LIB:-/home/thomas-ahle/z3-install/lib64/libz3.so}"
CIRCT_VERILOG="${CIRCT_VERILOG:-build/bin/circt-verilog}"
CIRCT_BMC="${CIRCT_BMC:-build/bin/circt-bmc}"
CIRCT_BMC_ARGS="${CIRCT_BMC_ARGS:-}"
BMC_SMOKE_ONLY="${BMC_SMOKE_ONLY:-0}"
KEEP_LOGS_DIR="${KEEP_LOGS_DIR:-}"
# NOTE: NO_PROPERTY_AS_SKIP defaults to 0 because the "no property provided to check"
# warning is SPURIOUS for clocked assertions that are lowered later in the pipeline.
# Setting this to 1 would cause false SKIP results for otherwise valid tests.
NO_PROPERTY_AS_SKIP="${NO_PROPERTY_AS_SKIP:-0}"
TOP="${TOP:-top}"
TEST_FILTER="${TEST_FILTER:-}"
OUT="${OUT:-$PWD/verilator-verification-bmc-results.txt}"
XFAILS="${XFAILS:-}"
DISABLE_UVM_AUTO_INCLUDE="${DISABLE_UVM_AUTO_INCLUDE:-1}"
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"

if [[ ! -d "$VERIF_DIR/tests" ]]; then
  echo "verilator-verification directory not found: $VERIF_DIR" >&2
  exit 1
fi

suites=("$@")
if [[ ${#suites[@]} -eq 0 ]]; then
  suites=("$VERIF_DIR/tests/asserts" "$VERIF_DIR/tests/sequences" \
    "$VERIF_DIR/tests/event-control-expression")
else
  for i in "${!suites[@]}"; do
    if [[ "${suites[$i]}" != /* ]]; then
      suites[$i]="$VERIF_DIR/${suites[$i]}"
    fi
  done
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
xfail=0
xpass=0
error=0
skip=0
total=0

is_xfail() {
  local name="$1"
  if [[ -z "$XFAILS" ]]; then
    return 1
  fi
  if [[ ! -f "$XFAILS" ]]; then
    return 1
  fi
  grep -Fxq "$name" "$XFAILS"
}

detect_top() {
  local file="$1"
  local requested="$2"
  local modules=()
  while IFS= read -r name; do
    modules+=("$name")
  done < <(awk '
    /^[[:space:]]*module[[:space:]]+/ {
      name=$2
      sub(/[^A-Za-z0-9_$].*/, "", name)
      if (name != "") print name
    }' "$file")

  if [[ -n "$requested" ]]; then
    for m in "${modules[@]}"; do
      if [[ "$m" == "$requested" ]]; then
        echo "$requested"
        return
      fi
    done
  fi

  if [[ ${#modules[@]} -eq 1 ]]; then
    echo "${modules[0]}"
    return
  fi

  if [[ -z "$requested" || "$requested" == "top" ]]; then
    if [[ ${#modules[@]} -gt 0 ]]; then
      echo "${modules[0]}"
      return
    fi
  fi

  echo "$requested"
}

for suite in "${suites[@]}"; do
  if [[ ! -d "$suite" ]]; then
    echo "suite not found: $suite" >&2
    continue
  fi
  while IFS= read -r -d '' sv; do
    base="$(basename "$sv" .sv)"
    if [[ -n "$TEST_FILTER" ]] && ! [[ "$base" =~ $TEST_FILTER ]]; then
      skip=$((skip + 1))
      continue
    fi

    log_tag="$base"
    rel_path="${sv#"$VERIF_DIR/"}"
    if [[ "$rel_path" != "$sv" ]]; then
      log_tag="${rel_path%.sv}"
    fi
    log_tag="${log_tag//\//__}"

    total=$((total + 1))

    mlir="$tmpdir/${base}.mlir"
    verilog_log="$tmpdir/${base}.circt-verilog.log"
    bmc_log="$tmpdir/${base}.circt-bmc.log"
    top_for_file="$(detect_top "$sv" "$TOP")"

    cmd=("$CIRCT_VERILOG" --ir-llhd --timescale=1ns/1ns --single-unit \
      -Wno-implicit-conv -Wno-index-oob -Wno-range-oob -Wno-range-width-oob)
    if [[ "$DISABLE_UVM_AUTO_INCLUDE" == "1" ]]; then
      cmd+=("--no-uvm-auto-include")
    fi
    if [[ -n "$CIRCT_VERILOG_ARGS" ]]; then
      read -r -a extra_args <<<"$CIRCT_VERILOG_ARGS"
      cmd+=("${extra_args[@]}")
    fi
    cmd+=("-I" "$(dirname "$sv")")
    if [[ -n "$top_for_file" ]]; then
      cmd+=("--top=$top_for_file")
    fi
    cmd+=("$sv")

    if ! "${cmd[@]}" > "$mlir" 2> "$verilog_log"; then
      result="ERROR"
      if is_xfail "$base"; then
        result="XFAIL"
        xfail=$((xfail + 1))
      else
        error=$((error + 1))
      fi
      printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
      continue
    fi

    bmc_args=("-b" "$BOUND" "--ignore-asserts-until=$IGNORE_ASSERTS_UNTIL" \
      "--module" "$top_for_file" "--shared-libs=$Z3_LIB")
    if [[ "$RISING_CLOCKS_ONLY" == "1" ]]; then
      bmc_args+=("--rising-clocks-only")
    fi
    if [[ "$ALLOW_MULTI_CLOCK" == "1" ]]; then
      bmc_args+=("--allow-multi-clock")
    fi
    if [[ -n "$CIRCT_BMC_ARGS" ]]; then
      read -r -a extra_bmc_args <<<"$CIRCT_BMC_ARGS"
      bmc_args+=("${extra_bmc_args[@]}")
    fi
    out=""
    if out="$("$CIRCT_BMC" "${bmc_args[@]}" "$mlir" 2> "$bmc_log")"; then
      bmc_status=0
    else
      bmc_status=$?
    fi
    if [[ "$NO_PROPERTY_AS_SKIP" == "1" ]] && \
        grep -q "no property provided to check in module" "$bmc_log"; then
      result="SKIP"
      skip=$((skip + 1))
      printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
      if [[ -n "$KEEP_LOGS_DIR" ]]; then
        mkdir -p "$KEEP_LOGS_DIR"
        cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
        cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
          2>/dev/null || true
        cp -f "$bmc_log" "$KEEP_LOGS_DIR/${log_tag}.circt-bmc.log" \
          2>/dev/null || true
      fi
      continue
    fi

    if [[ "$BMC_SMOKE_ONLY" == "1" ]]; then
      if [[ "$bmc_status" -eq 0 ]]; then
        result="PASS"
      else
        result="ERROR"
      fi
    else
      if grep -q "Bound reached with no violations!" <<<"$out"; then
        result="PASS"
      elif grep -q "Assertion can be violated!" <<<"$out"; then
        result="FAIL"
      else
        result="ERROR"
      fi
    fi

    if [[ "$BMC_SMOKE_ONLY" == "1" ]] && is_xfail "$base"; then
      result="XFAIL"
      xfail=$((xfail + 1))
      printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
      if [[ -n "$KEEP_LOGS_DIR" ]]; then
        mkdir -p "$KEEP_LOGS_DIR"
        cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
        cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
          2>/dev/null || true
        cp -f "$bmc_log" "$KEEP_LOGS_DIR/${log_tag}.circt-bmc.log" \
          2>/dev/null || true
      fi
      continue
    fi

    if is_xfail "$base"; then
      if [[ "$result" == "PASS" ]]; then
        result="XPASS"
        xpass=$((xpass + 1))
      else
        result="XFAIL"
        xfail=$((xfail + 1))
      fi
    else
      case "$result" in
        PASS) pass=$((pass + 1)) ;;
        FAIL) fail=$((fail + 1)) ;;
        *) error=$((error + 1)) ;;
      esac
    fi

    printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
    if [[ -n "$KEEP_LOGS_DIR" ]]; then
      mkdir -p "$KEEP_LOGS_DIR"
      cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
      cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
        2>/dev/null || true
      cp -f "$bmc_log" "$KEEP_LOGS_DIR/${log_tag}.circt-bmc.log" \
        2>/dev/null || true
    fi
  done < <(find "$suite" -type f -name "*.sv" -print0)
done

sort "$results_tmp" > "$OUT"

echo "verilator-verification summary: total=$total pass=$pass fail=$fail xfail=$xfail xpass=$xpass error=$error skip=$skip"
echo "results: $OUT"
