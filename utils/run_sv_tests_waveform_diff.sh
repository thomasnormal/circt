#!/usr/bin/env bash
# Run sv-tests simulation cases across 4 lanes and compare waveforms:
#   1) interpreted (fast path OFF)
#   2) interpreted (fast path ON)
#   3) compile/AOT mode
#   4) xcelium
#
# Produces lane matrices and pairwise waveform parity TSVs.
#
# Usage:
#   utils/run_sv_tests_waveform_diff.sh [sv_tests_dir] [out_dir]
#
# Key env vars:
#   TEST_FILTER=regex
#   TAG_REGEX=regex
#   MAX_TESTS=N
#   SKIP_UVM=0|1 (default: 0)
#   MAX_SIM_TIME=10000000000000
#   CIRCT_COMPILE_TIMEOUT_SECS=120
#   CIRCT_SIM_TIMEOUT_SECS=30
#   XCELIUM_SIM_TIMEOUT_SECS=30
#   XCELIUM_RUN_TIME=10us
#   CIRCT_SIM_TRACE_ALL=1
#   CIRCT_SIM_EXTRA_ARGS="..."
#   WAVE_COMPARE_ARGS="--require-same-signal-set"
#   COMPARE_NONFUNCTIONAL=0|1
#   FAIL_ON_WAVE_MISMATCH=0|1
#   FAIL_ON_MISSING_VCD=0|1
#   REQUIRE_XCELIUM=0|1

set -euo pipefail

SV_TESTS_DIR="${1:-/home/thomas-ahle/sv-tests}"
OUT_DIR="${2:-/tmp/sv-tests-waveform-diff-$(date +%Y%m%d-%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEST_FILTER="${TEST_FILTER:-}"
TAG_REGEX="${TAG_REGEX:-}"
MAX_TESTS="${MAX_TESTS:-0}"
SKIP_UVM="${SKIP_UVM:-0}"
MAX_SIM_TIME="${MAX_SIM_TIME:-10000000000000}"
CIRCT_COMPILE_TIMEOUT_SECS="${CIRCT_COMPILE_TIMEOUT_SECS:-120}"
CIRCT_SIM_TIMEOUT_SECS="${CIRCT_SIM_TIMEOUT_SECS:-30}"
XCELIUM_SIM_TIMEOUT_SECS="${XCELIUM_SIM_TIMEOUT_SECS:-30}"
XCELIUM_RUN_TIME="${XCELIUM_RUN_TIME:-10us}"
CIRCT_SIM_TRACE_ALL="${CIRCT_SIM_TRACE_ALL:-1}"
CIRCT_SIM_EXTRA_ARGS="${CIRCT_SIM_EXTRA_ARGS:-}"
DISABLE_UVM_AUTO_INCLUDE="${DISABLE_UVM_AUTO_INCLUDE:-1}"

WAVE_COMPARE_ARGS="${WAVE_COMPARE_ARGS:---require-same-signal-set --ignore-timescale}"
COMPARE_NONFUNCTIONAL="${COMPARE_NONFUNCTIONAL:-0}"
FAIL_ON_WAVE_MISMATCH="${FAIL_ON_WAVE_MISMATCH:-0}"
FAIL_ON_MISSING_VCD="${FAIL_ON_MISSING_VCD:-0}"
REQUIRE_XCELIUM="${REQUIRE_XCELIUM:-1}"

CIRCT_VERILOG="${CIRCT_VERILOG:-$PWD/build_test/bin/circt-verilog}"
CIRCT_SIM="${CIRCT_SIM:-$PWD/build_test/bin/circt-sim}"
XRUN="${XRUN:-$(command -v xrun || true)}"
COMPARE_VCD_TOOL="${COMPARE_VCD_TOOL:-$SCRIPT_DIR/compare_vcd_waveforms.py}"
CHECK_WAVE_PARITY="${CHECK_WAVE_PARITY:-$SCRIPT_DIR/check_avip_waveform_matrix_parity.py}"

if [[ ! -d "$SV_TESTS_DIR/tests" ]]; then
  echo "error: sv-tests directory not found: $SV_TESTS_DIR" >&2
  exit 1
fi

wait_for_executable() {
  local tool="$1"
  local label="$2"
  local tries=10
  local i=0
  for ((i = 1; i <= tries; ++i)); do
    if [[ -x "$tool" ]]; then
      return 0
    fi
    sleep 1
  done
  echo "error: $label not executable after ${tries}s: $tool" >&2
  return 1
}

wait_for_executable "$CIRCT_VERILOG" "CIRCT_VERILOG"
wait_for_executable "$CIRCT_SIM" "CIRCT_SIM"
if [[ ! -f "$COMPARE_VCD_TOOL" ]]; then
  echo "error: compare tool not found: $COMPARE_VCD_TOOL" >&2
  exit 1
fi
if [[ ! -f "$CHECK_WAVE_PARITY" ]]; then
  echo "error: wave parity tool not found: $CHECK_WAVE_PARITY" >&2
  exit 1
fi
if [[ "$REQUIRE_XCELIUM" != "0" && -z "$XRUN" ]]; then
  echo "error: xrun not found (set XRUN or REQUIRE_XCELIUM=0)" >&2
  exit 1
fi
if [[ "$MAX_TESTS" != "0" && ! "$MAX_TESTS" =~ ^[0-9]+$ ]]; then
  echo "error: MAX_TESTS must be 0 or a non-negative integer" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$OUT_DIR/cases" "$OUT_DIR/lanes"

# Snapshot CIRCT tools once per run to avoid concurrent rebuild races.
TOOL_SNAPSHOT_DIR="$OUT_DIR/.tool-snapshot"
mkdir -p "$TOOL_SNAPSHOT_DIR"
SNAPSHOT_CIRCT_VERILOG="$TOOL_SNAPSHOT_DIR/circt-verilog"
SNAPSHOT_CIRCT_SIM="$TOOL_SNAPSHOT_DIR/circt-sim"
cp -f "$CIRCT_VERILOG" "$SNAPSHOT_CIRCT_VERILOG"
cp -f "$CIRCT_SIM" "$SNAPSHOT_CIRCT_SIM"
chmod +x "$SNAPSHOT_CIRCT_VERILOG" "$SNAPSHOT_CIRCT_SIM" 2>/dev/null || true
CIRCT_VERILOG="$SNAPSHOT_CIRCT_VERILOG"
CIRCT_SIM="$SNAPSHOT_CIRCT_SIM"

echo "[sv-tests-wave] out_dir=$OUT_DIR"
echo "[sv-tests-wave] sv_tests_dir=$SV_TESTS_DIR"

LANES=(interpret_off interpret_on compile_aot xcelium)
for lane in "${LANES[@]}"; do
  lane_dir="$OUT_DIR/lanes/$lane"
  mkdir -p "$lane_dir"
  cat > "$lane_dir/matrix.tsv" <<'HDR'
avip	seed	compile_status	compile_sec	sim_status	sim_exit	sim_sec	sim_time_fs	uvm_fatal	uvm_error	cov_1_pct	cov_2_pct	peak_rss_kb	compile_log	sim_log	vcd_file
HDR
done

read_meta() {
  local key="$1"
  local file="$2"
  sed -n "s/^[[:space:]]*:${key}:[[:space:]]*//p" "$file" | head -n 1
}

normalize_paths() {
  local root="$1"
  shift
  local out=()
  local item=""
  for item in "$@"; do
    [[ -z "$item" ]] && continue
    if [[ "$item" == /* ]]; then
      out+=("$item")
    else
      out+=("$root/$item")
    fi
  done
  printf '%s\n' "${out[@]}"
}

sim_status_from_exit() {
  local code="$1"
  case "$code" in
    0) echo "OK" ;;
    124|137) echo "TIMEOUT" ;;
    *) echo "FAIL" ;;
  esac
}

run_limited_with_timeout() {
  local timeout_secs="$1"
  shift
  timeout --signal=KILL "$timeout_secs" "$@"
}

append_lane_row() {
  local lane="$1"
  local case_id="$2"
  local compile_status="$3"
  local compile_sec="$4"
  local sim_status="$5"
  local sim_exit="$6"
  local sim_sec="$7"
  local compile_log="$8"
  local sim_log="$9"
  local vcd_file="${10}"
  local matrix="$OUT_DIR/lanes/$lane/matrix.tsv"

  printf "%s\t1\t%s\t%s\t%s\t%s\t%s\t-\t0\t0\t-\t-\t-\t%s\t%s\t%s\n" \
    "$case_id" "$compile_status" "$compile_sec" "$sim_status" "$sim_exit" "$sim_sec" \
    "$compile_log" "$sim_log" "$vcd_file" >> "$matrix"
}

run_circt_lane() {
  local lane="$1"
  local case_dir="$2"
  local mlir="$3"
  local top_module="$4"
  local mode="$5"
  local fastpaths="$6"
  local compile_status="$7"
  local compile_sec="$8"
  local compile_log="$9"

  local case_id
  case_id="$(basename "$case_dir")"
  local lane_dir="$case_dir/$lane"
  mkdir -p "$lane_dir"

  local sim_log="$lane_dir/sim.log"
  local vcd_file="$lane_dir/waves.vcd"

  if [[ "$compile_status" != "OK" ]]; then
    append_lane_row "$lane" "$case_id" "$compile_status" "$compile_sec" "SKIP" "-" "-" "$compile_log" "$sim_log" "-"
    return 0
  fi

  local cmd=("$CIRCT_SIM" "$mlir" "--top=$top_module" "--mode=$mode" "--max-time=$MAX_SIM_TIME" "--vcd=$vcd_file")
  if [[ "$CIRCT_SIM_TRACE_ALL" != "0" ]]; then
    cmd+=(--trace-all)
  fi
  if [[ -n "$CIRCT_SIM_EXTRA_ARGS" ]]; then
    local extra=()
    read -r -a extra <<< "$CIRCT_SIM_EXTRA_ARGS"
    cmd+=("${extra[@]}")
  fi

  local sim_exit=0
  local start end sim_sec sim_status
  start="$(date +%s)"
  set +e
  CIRCT_SIM_ENABLE_DIRECT_FASTPATHS="$fastpaths" \
    run_limited_with_timeout "$CIRCT_SIM_TIMEOUT_SECS" "${cmd[@]}" > "$sim_log" 2>&1
  sim_exit=$?
  set -e
  end="$(date +%s)"
  sim_sec="$((end - start))"
  sim_status="$(sim_status_from_exit "$sim_exit")"

  if [[ ! -s "$vcd_file" ]]; then
    vcd_file="-"
  fi
  append_lane_row "$lane" "$case_id" "$compile_status" "$compile_sec" "$sim_status" "$sim_exit" "$sim_sec" "$compile_log" "$sim_log" "$vcd_file"
}

run_xcelium_lane() {
  local case_dir="$1"
  local top_module="$2"
  local top_from_meta="$3"
  shift 3
  local files=("$@")

  local case_id
  case_id="$(basename "$case_dir")"
  local lane="xcelium"
  local lane_dir="$case_dir/$lane"
  mkdir -p "$lane_dir"

  local sim_log="$lane_dir/sim.log"
  local compile_log="$lane_dir/compile.log"
  local vcd_file="$lane_dir/waves.vcd"
  local wave_tcl="$lane_dir/wave.tcl"
  local wave_tcl_name="wave.tcl"
  local probe_target="$top_module"
  if [[ -z "$probe_target" ]]; then
    probe_target="top"
  fi

  if [[ -z "$XRUN" ]]; then
    append_lane_row "$lane" "$case_id" "SKIP" "-" "SKIP" "-" "-" "$compile_log" "$sim_log" "-"
    return 0
  fi

  cat > "$wave_tcl" <<TCL
database -open waves -vcd -into waves.vcd
probe -create $probe_target -all -depth all
run $XCELIUM_RUN_TIME
exit
TCL

  local cmd=("$XRUN" -64bit -sv -access +rwc "-input" "$wave_tcl_name")
  local inc
  for inc in "${XCELIUM_INCDIRS[@]}"; do
    cmd+=("+incdir+$inc")
  done
  local def
  for def in "${XCELIUM_DEFINES[@]}"; do
    cmd+=("+define+$def")
  done
  if [[ -n "$top_module" ]]; then
    cmd+=("-top" "$top_module")
  fi
  cmd+=("${files[@]}")

  rm -f "$vcd_file"
  local sim_exit=0
  local start end sim_sec sim_status
  start="$(date +%s)"
  set +e
  (
    cd "$lane_dir"
    run_limited_with_timeout "$XCELIUM_SIM_TIMEOUT_SECS" "${cmd[@]}"
  ) > "$sim_log" 2>&1
  sim_exit=$?
  set -e

  # Retry without explicit -top only when top came from default inference.
  if [[ "$sim_exit" -ne 0 && -z "$top_from_meta" ]]; then
    local cmd_notop=("$XRUN" -64bit -sv -access +rwc "-input" "$wave_tcl_name")
    for inc in "${XCELIUM_INCDIRS[@]}"; do
      cmd_notop+=("+incdir+$inc")
    done
    for def in "${XCELIUM_DEFINES[@]}"; do
      cmd_notop+=("+define+$def")
    done
    cmd_notop+=("${files[@]}")
    set +e
    (
      cd "$lane_dir"
      run_limited_with_timeout "$XCELIUM_SIM_TIMEOUT_SECS" "${cmd_notop[@]}"
    ) > "$sim_log" 2>&1
    sim_exit=$?
    set -e
  fi

  end="$(date +%s)"
  sim_sec="$((end - start))"
  sim_status="$(sim_status_from_exit "$sim_exit")"

  cp -f "$sim_log" "$compile_log" 2>/dev/null || true

  if [[ ! -s "$vcd_file" ]]; then
    vcd_file="-"
  fi
  append_lane_row "$lane" "$case_id" "OK" "-" "$sim_status" "$sim_exit" "$sim_sec" "$compile_log" "$sim_log" "$vcd_file"
}

selected=0
skipped=0
max_tests_reached=0

test_root="$SV_TESTS_DIR/tests"
mapfile -d '' sv_candidates < <(find "$SV_TESTS_DIR/tests" -type f -name "*.sv" -print0 | sort -z)
for sv in "${sv_candidates[@]}"; do
  tags="$(read_meta tags "$sv")"
  type="$(read_meta type "$sv")"
  should_fail="$(read_meta should_fail "$sv")"
  should_fail_because="$(read_meta should_fail_because "$sv")"
  base="$(basename "$sv" .sv)"

  if [[ -n "$type" ]] && ! [[ "$type" =~ [Ss]imulation ]]; then
    skipped=$((skipped + 1))
    continue
  fi

  if [[ "$should_fail" == "1" || -n "$should_fail_because" ]]; then
    skipped=$((skipped + 1))
    continue
  fi

  if [[ -n "$TAG_REGEX" ]]; then
    if [[ -z "$tags" ]] || ! [[ "$tags" =~ $TAG_REGEX ]]; then
      skipped=$((skipped + 1))
      continue
    fi
  fi

  if [[ "$SKIP_UVM" != "0" ]]; then
    if [[ "$tags" =~ [Uu][Vv][Mm] ]] || [[ "$base" =~ [Uu][Vv][Mm] ]]; then
      skipped=$((skipped + 1))
      continue
    fi
  fi

  if [[ -n "$TEST_FILTER" ]] && ! [[ "$base" =~ $TEST_FILTER ]]; then
    skipped=$((skipped + 1))
    continue
  fi

  if [[ "$MAX_TESTS" != "0" && "$selected" -ge "$MAX_TESTS" ]]; then
    max_tests_reached=1
    break
  fi
  selected=$((selected + 1))

  rel_path="${sv#"$test_root/"}"
  case_id="${rel_path%.sv}"
  case_id="${case_id//\//__}"
  case_dir="$OUT_DIR/cases/$case_id"
  mkdir -p "$case_dir"

  files_line="$(read_meta files "$sv")"
  incdirs_line="$(read_meta incdirs "$sv")"
  defines_line="$(read_meta defines "$sv")"
  top_module="$(read_meta top_module "$sv")"
  top_from_meta="$top_module"
  if [[ -z "$top_module" ]]; then
    top_module="top"
  fi

  files=()
  if [[ -z "$files_line" ]]; then
    files=("$sv")
  else
    mapfile -t files < <(normalize_paths "$test_root" $files_line)
  fi

  incdirs=()
  if [[ -n "$incdirs_line" ]]; then
    mapfile -t incdirs < <(normalize_paths "$test_root" $incdirs_line)
  fi
  incdirs+=("$(dirname "$sv")")

  defines=()
  if [[ -n "$defines_line" ]]; then
    for d in $defines_line; do
      defines+=("$d")
    done
  fi

  XCELIUM_INCDIRS=("${incdirs[@]}")
  XCELIUM_DEFINES=("${defines[@]}")

  mlir="$case_dir/case.mlir"
  compile_log="$case_dir/circt-verilog.log"
  compile_status="FAIL"
  compile_sec="-"

  cmd=("$CIRCT_VERILOG" --ir-llhd --timescale=1ns/1ns --single-unit \
    -Wno-implicit-conv -Wno-index-oob -Wno-range-oob -Wno-range-width-oob)
  if [[ "$DISABLE_UVM_AUTO_INCLUDE" == "1" ]]; then
    cmd+=(--no-uvm-auto-include)
  fi
  local_inc=""
  for local_inc in "${incdirs[@]}"; do
    cmd+=("-I" "$local_inc")
  done
  local_def=""
  for local_def in "${defines[@]}"; do
    cmd+=("-D" "$local_def")
  done
  if [[ -n "$top_module" ]]; then
    cmd+=("--top=$top_module")
  fi
  cmd+=("${files[@]}")

  start_compile="$(date +%s)"
  if run_limited_with_timeout "$CIRCT_COMPILE_TIMEOUT_SECS" "${cmd[@]}" > "$mlir" 2> "$compile_log"; then
    compile_status="OK"
  elif [[ -z "$top_from_meta" ]] && grep -qE "is not a valid top-level module|could not resolve hierarchical path" "$compile_log" 2>/dev/null; then
    cmd_notop=("$CIRCT_VERILOG" --ir-llhd --timescale=1ns/1ns --single-unit \
      -Wno-implicit-conv -Wno-index-oob -Wno-range-oob -Wno-range-width-oob)
    if [[ "$DISABLE_UVM_AUTO_INCLUDE" == "1" ]]; then
      cmd_notop+=(--no-uvm-auto-include)
    fi
    for local_inc in "${incdirs[@]}"; do
      cmd_notop+=("-I" "$local_inc")
    done
    for local_def in "${defines[@]}"; do
      cmd_notop+=("-D" "$local_def")
    done
    cmd_notop+=("${files[@]}")
    if run_limited_with_timeout "$CIRCT_COMPILE_TIMEOUT_SECS" "${cmd_notop[@]}" > "$mlir" 2> "$compile_log"; then
      compile_status="OK"
      inferred_top="$(
        grep -m1 'llhd.entity\|hw.module' "$mlir" \
          | sed -n 's/.*@\([a-zA-Z_][a-zA-Z0-9_]*\).*/\1/p' \
          || true
      )"
      if [[ -n "$inferred_top" ]]; then
        top_module="$inferred_top"
      fi
    fi
  fi
  end_compile="$(date +%s)"
  compile_sec="$((end_compile - start_compile))"

  if [[ "$compile_status" == "OK" ]] && ! grep -q 'llhd.entity\|llhd.proc\|hw.module' "$mlir"; then
    compile_status="NO_TOP"
  fi

  echo "[sv-tests-wave] case=$case_id compile_status=$compile_status top=$top_module"

  run_circt_lane interpret_off "$case_dir" "$mlir" "$top_module" interpret 0 "$compile_status" "$compile_sec" "$compile_log"
  run_circt_lane interpret_on  "$case_dir" "$mlir" "$top_module" interpret 1 "$compile_status" "$compile_sec" "$compile_log"
  run_circt_lane compile_aot   "$case_dir" "$mlir" "$top_module" compile   0 "$compile_status" "$compile_sec" "$compile_log"

  run_xcelium_lane "$case_dir" "$top_module" "$top_from_meta" "${files[@]}"
done

if [[ "$selected" -eq 0 ]]; then
  echo "error: no sv-tests selected (check TEST_FILTER/TAG_REGEX)" >&2
  exit 1
fi

compare_pair() {
  local lhs="$1"
  local rhs="$2"
  local out_tsv="$3"

  local cmd=(
    python3 "$CHECK_WAVE_PARITY"
    --lhs-matrix "$OUT_DIR/lanes/$lhs/matrix.tsv"
    --rhs-matrix "$OUT_DIR/lanes/$rhs/matrix.tsv"
    --lhs-label "$lhs"
    --rhs-label "$rhs"
    --compare-tool "$COMPARE_VCD_TOOL"
    --out-tsv "$out_tsv"
  )

  if [[ -n "$WAVE_COMPARE_ARGS" ]]; then
    cmd+=("--compare-arg=$WAVE_COMPARE_ARGS")
  fi
  if [[ "$COMPARE_NONFUNCTIONAL" != "0" ]]; then
    cmd+=(--compare-nonfunctional)
  fi
  if [[ "$FAIL_ON_WAVE_MISMATCH" != "0" ]]; then
    cmd+=(--fail-on-mismatch)
  fi
  if [[ "$FAIL_ON_MISSING_VCD" != "0" ]]; then
    cmd+=(--fail-on-missing-vcd)
  fi

  "${cmd[@]}"
}

compare_pair interpret_off interpret_on "$OUT_DIR/parity_interpret_off_vs_interpret_on.tsv"
compare_pair interpret_off compile_aot "$OUT_DIR/parity_interpret_off_vs_compile_aot.tsv"
compare_pair interpret_off xcelium "$OUT_DIR/parity_interpret_off_vs_xcelium.tsv"

count_lane_vcd_rows() {
  local matrix="$1"
  awk -F'\t' '
    NR == 1 {
      for (i = 1; i <= NF; ++i)
        if ($i == "vcd_file")
          col = i
      next
    }
    col > 0 {
      v = $col
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
      if (v != "" && v != "-" && v != "?")
        n += 1
    }
    END { print n + 0 }
  ' "$matrix"
}

vcd_interpret_off="$(count_lane_vcd_rows "$OUT_DIR/lanes/interpret_off/matrix.tsv")"
vcd_interpret_on="$(count_lane_vcd_rows "$OUT_DIR/lanes/interpret_on/matrix.tsv")"
vcd_compile_aot="$(count_lane_vcd_rows "$OUT_DIR/lanes/compile_aot/matrix.tsv")"
vcd_xcelium="$(count_lane_vcd_rows "$OUT_DIR/lanes/xcelium/matrix.tsv")"
waves_upper_bound=$((selected * ${#LANES[@]}))
waves_produced=$((vcd_interpret_off + vcd_interpret_on + vcd_compile_aot + vcd_xcelium))

echo "[sv-tests-wave] selected=$selected skipped=$skipped"
echo "[sv-tests-wave] max_tests_reached=$max_tests_reached"
echo "[sv-tests-wave] waves_upper_bound=$waves_upper_bound waves_produced=$waves_produced"
echo "[sv-tests-wave] lane_vcd_counts=interpret_off:$vcd_interpret_off interpret_on:$vcd_interpret_on compile_aot:$vcd_compile_aot xcelium:$vcd_xcelium"
echo "[sv-tests-wave] lane_matrix_interpret_off=$OUT_DIR/lanes/interpret_off/matrix.tsv"
echo "[sv-tests-wave] lane_matrix_interpret_on=$OUT_DIR/lanes/interpret_on/matrix.tsv"
echo "[sv-tests-wave] lane_matrix_compile_aot=$OUT_DIR/lanes/compile_aot/matrix.tsv"
echo "[sv-tests-wave] lane_matrix_xcelium=$OUT_DIR/lanes/xcelium/matrix.tsv"
echo "[sv-tests-wave] parity_off_vs_on=$OUT_DIR/parity_interpret_off_vs_interpret_on.tsv"
echo "[sv-tests-wave] parity_off_vs_aot=$OUT_DIR/parity_interpret_off_vs_compile_aot.tsv"
echo "[sv-tests-wave] parity_off_vs_xcelium=$OUT_DIR/parity_interpret_off_vs_xcelium.tsv"
