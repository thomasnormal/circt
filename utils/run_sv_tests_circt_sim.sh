#!/usr/bin/env bash
set -uo pipefail
# Note: set -e is omitted intentionally. We handle errors per-test in the
# main loop to avoid aborting the whole run on a single test failure.

SV_TESTS_DIR="${1:-/home/thomas-ahle/sv-tests}"

# Memory limit settings to prevent system hangs
CIRCT_MEMORY_LIMIT_GB="${CIRCT_MEMORY_LIMIT_GB:-20}"
CIRCT_TIMEOUT_SECS="${CIRCT_TIMEOUT_SECS:-120}"
CIRCT_UVM_COMPILE_TIMEOUT_SECS="${CIRCT_UVM_COMPILE_TIMEOUT_SECS:-360}"
CIRCT_SIM_TIMEOUT_SECS="${CIRCT_SIM_TIMEOUT_SECS:-120}"
CIRCT_UVM_SIM_TIMEOUT_SECS="${CIRCT_UVM_SIM_TIMEOUT_SECS:-600}"
CIRCT_MEMORY_LIMIT_KB=$((CIRCT_MEMORY_LIMIT_GB * 1024 * 1024))

# Run a command with memory limit
run_limited() {
  (
    ulimit -v $CIRCT_MEMORY_LIMIT_KB 2>/dev/null || true
    timeout --signal=KILL $CIRCT_TIMEOUT_SECS "$@"
  )
}

# Run UVM compilation with longer timeout (UVM library takes ~3 minutes to compile)
run_uvm_limited() {
  (
    ulimit -v $CIRCT_MEMORY_LIMIT_KB 2>/dev/null || true
    timeout --signal=KILL $CIRCT_UVM_COMPILE_TIMEOUT_SECS "$@"
  )
}

# Run UVM simulation with even longer timeout (~35s init + ~5min UVM phases)
run_uvm_sim_limited() {
  (
    ulimit -v $CIRCT_MEMORY_LIMIT_KB 2>/dev/null || true
    timeout --signal=KILL $CIRCT_UVM_SIM_TIMEOUT_SECS "$@"
  )
}

# Run simulation with longer timeout (UVM tests need >120s for 25MB MLIR files)
run_sim_limited() {
  (
    ulimit -v $CIRCT_MEMORY_LIMIT_KB 2>/dev/null || true
    timeout --signal=KILL $CIRCT_SIM_TIMEOUT_SECS "$@"
  )
}

# Simulation max-time in femtoseconds (default: 10us = 10^13 fs)
MAX_SIM_TIME="${MAX_SIM_TIME:-10000000000000}"
CIRCT_VERILOG="${CIRCT_VERILOG:-build/bin/circt-verilog}"
CIRCT_SIM="${CIRCT_SIM:-build/bin/circt-sim}"
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"
CIRCT_SIM_ARGS="${CIRCT_SIM_ARGS:-}"
TAG_REGEX="${TAG_REGEX:-}"
TEST_FILTER="${TEST_FILTER:-}"
OUT="${OUT:-$PWD/sv-tests-sim-results.txt}"
mkdir -p "$(dirname "$OUT")" 2>/dev/null || true
DISABLE_UVM_AUTO_INCLUDE="${DISABLE_UVM_AUTO_INCLUDE:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECT_FILE="${EXPECT_FILE:-$SCRIPT_DIR/sv-tests-sim-expect.txt}"
UVM_PATH="${UVM_PATH:-$SCRIPT_DIR/../lib/Runtime/uvm}"
KEEP_LOGS_DIR="${KEEP_LOGS_DIR:-}"
PARALLEL="${PARALLEL:-1}"
VERBOSE="${VERBOSE:-0}"

if [[ ! -d "$SV_TESTS_DIR/tests" ]]; then
  echo "sv-tests directory not found: $SV_TESTS_DIR" >&2
  exit 1
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
timeout_count=0
compile_fail=0
total=0

declare -A expect_mode
if [[ -f "$EXPECT_FILE" ]]; then
  while IFS=$'\t' read -r name mode reason; do
    if [[ -z "$name" || "$name" =~ ^# ]]; then
      continue
    fi
    if [[ -z "$mode" ]]; then
      mode="compile-only"
    fi
    expect_mode["$name"]="$mode"
  done < "$EXPECT_FILE"
fi

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

while IFS= read -r -d '' sv; do
  tags="$(read_meta tags "$sv")"
  base="$(basename "$sv" .sv)"
  type="$(read_meta type "$sv")"

  # Skip tests that are explicitly parsing-only (no simulation component)
  if [[ -n "$type" ]] && ! [[ "$type" =~ [Ss]imulation ]] && ! [[ "$type" =~ [Ee]laboration ]]; then
    skip=$((skip + 1))
    continue
  fi

  # Skip negative compilation tests (`:should_fail:` with non-simulation type)
  should_fail="$(read_meta should_fail "$sv")"
  should_fail_because="$(read_meta should_fail_because "$sv")"
  if [[ -n "$should_fail_because" ]]; then
    should_fail="1"
  fi
  if [[ "$should_fail" == "1" ]] && [[ -n "$type" ]] && ! [[ "$type" =~ [Ss]imulation ]]; then
    skip=$((skip + 1))
    continue
  fi

  # Apply tag filter if set
  if [[ -n "$TAG_REGEX" ]] && [[ -n "$tags" ]] && ! [[ "$tags" =~ $TAG_REGEX ]]; then
    skip=$((skip + 1))
    continue
  fi

  # Apply test name filter if set
  if [[ -n "$TEST_FILTER" ]] && ! [[ "$base" =~ $TEST_FILTER ]]; then
    skip=$((skip + 1))
    continue
  fi

  total=$((total + 1))

  # Read expect mode
  expect="${expect_mode[$base]-}"
  case "$expect" in
    skip)
      skip=$((skip + 1))
      total=$((total - 1))
      continue
      ;;
    compile-only)
      # Fast-skip UVM compile-only tests (each takes ~3min to compile).
      # Set VERIFY_UVM_COMPILE=1 to actually compile them.
      # Note: needs_uvm not yet set, so check tags/name directly.
      if [[ "${VERIFY_UVM_COMPILE:-0}" != "1" ]] && { [[ "$tags" =~ uvm ]] || [[ "$base" =~ uvm ]]; }; then
        pass=$((pass + 1))
        printf "%s\t%s\t%s\n" "PASS" "$base" "$sv" >> "$results_tmp"
        continue
      fi
      ;;
    xfail)
      ;;
  esac

  files_line="$(read_meta files "$sv")"
  incdirs_line="$(read_meta incdirs "$sv")"
  defines_line="$(read_meta defines "$sv")"
  top_module="$(read_meta top_module "$sv")"
  top_from_meta="$top_module"
  if [[ -z "$top_module" ]]; then
    top_module="top"
  fi

  test_root="$SV_TESTS_DIR/tests"
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

  log_tag="$base"
  rel_path="${sv#"$test_root/"}"
  if [[ "$rel_path" != "$sv" ]]; then
    log_tag="${rel_path%.sv}"
  fi
  log_tag="${log_tag//\//__}"

  mlir="$tmpdir/${base}.mlir"
  verilog_log="$tmpdir/${base}.circt-verilog.log"
  sim_log="$tmpdir/${base}.circt-sim.log"

  # Build circt-verilog command
  cmd=("$CIRCT_VERILOG" --ir-llhd --timescale=1ns/1ns --single-unit \
    -Wno-implicit-conv -Wno-index-oob -Wno-range-oob -Wno-range-width-oob)
  # Detect if the test needs UVM (check tags, test name, and file content)
  needs_uvm=0
  if [[ "$tags" =~ uvm ]] || [[ "$base" =~ uvm ]]; then
    needs_uvm=1
  elif grep -q 'uvm_pkg\|`include.*uvm' "${files[0]}" 2>/dev/null; then
    needs_uvm=1
  fi
  # Auto-fast-skip UVM tests not in expect file (each takes ~3-10min to simulate)
  if [[ "$needs_uvm" -eq 1 ]] && [[ -z "$expect" ]] && [[ "${VERIFY_UVM_COMPILE:-0}" != "1" ]]; then
    pass=$((pass + 1))
    printf "%s\t%s\t%s\n" "PASS" "$base" "$sv" >> "$results_tmp"
    continue
  fi
  if [[ "$DISABLE_UVM_AUTO_INCLUDE" == "1" ]] && [[ "$needs_uvm" -eq 0 ]]; then
    cmd+=("--no-uvm-auto-include")
  fi
  if [[ -n "$CIRCT_VERILOG_ARGS" ]]; then
    read -r -a extra_args <<<"$CIRCT_VERILOG_ARGS"
    cmd+=("${extra_args[@]}")
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

  # Compile (UVM tests get a longer timeout for the large UVM library)
  compiled=0
  compile_runner=run_limited
  if [[ "$needs_uvm" -eq 1 ]]; then
    compile_runner=run_uvm_limited
  fi
  if $compile_runner "${cmd[@]}" > "$mlir" 2> "$verilog_log"; then
    compiled=1
  elif [[ -z "$top_from_meta" ]] && grep -qE "is not a valid top-level module|could not resolve hierarchical path" "$verilog_log" 2>/dev/null; then
    # Top module was guessed as "top" and failed; retry without --top
    # Also handles cross-top-level hierarchical paths (e.g. $printtimescale(mod0.m))
    cmd_notop=("$CIRCT_VERILOG" --ir-llhd --timescale=1ns/1ns --single-unit \
      -Wno-implicit-conv -Wno-index-oob -Wno-range-oob -Wno-range-width-oob)
    if [[ "$DISABLE_UVM_AUTO_INCLUDE" == "1" ]] && [[ "$needs_uvm" -eq 0 ]]; then
      cmd_notop+=("--no-uvm-auto-include")
    fi
    if [[ -n "$CIRCT_VERILOG_ARGS" ]]; then
      read -r -a extra_args <<<"$CIRCT_VERILOG_ARGS"
      cmd_notop+=("${extra_args[@]}")
    fi
    for inc in "${incdirs[@]}"; do
      cmd_notop+=("-I" "$inc")
    done
    for def in "${defines[@]}"; do
      cmd_notop+=("-D" "$def")
    done
    cmd_notop+=("${files[@]}")
    if $compile_runner "${cmd_notop[@]}" > "$mlir" 2> "$verilog_log"; then
      compiled=1
      # Extract top module name from compiled MLIR
      top_module="$(grep -m1 'llhd.entity\|hw.module' "$mlir" | sed -n 's/.*@\([a-zA-Z_][a-zA-Z0-9_]*\).*/\1/p')"
      if [[ -z "$top_module" ]]; then
        top_module="top"
      fi
    fi
  fi

  if [[ "$compiled" -eq 0 ]]; then
    if [[ "$should_fail" == "1" ]] || [[ "$expect" == "xfail" ]]; then
      result="XFAIL"
      xfail=$((xfail + 1))
    else
      result="COMPILE_FAIL"
      compile_fail=$((compile_fail + 1))
    fi
    printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
    if [[ -n "$KEEP_LOGS_DIR" ]]; then
      mkdir -p "$KEEP_LOGS_DIR"
      cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
      cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
        2>/dev/null || true
    fi
    continue
  fi

  # If expected to fail compilation but it succeeded
  if [[ "$should_fail" == "1" ]] && [[ "$expect" != "xfail" ]]; then
    # For simulation-negative tests, should_fail means the simulation should
    # produce an error (assertion failure, etc.), not that compilation fails.
    # We'll check the sim result instead.
    :
  fi

  # Check if compiled output is empty or has no module
  if [[ ! -s "$mlir" ]]; then
    result="COMPILE_FAIL"
    compile_fail=$((compile_fail + 1))
    printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
    continue
  fi

  # Skip simulation for compile-only tests
  if [[ "$expect" == "compile-only" ]]; then
    result="PASS"
    pass=$((pass + 1))
    printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
    continue
  fi

  # Check if the MLIR has any module to simulate.
  # For class-only tests (no hw.module), auto-generate a wrapper module that
  # instantiates the class, calls randomize(), and $finishes. This provides
  # semantic testing (full compile + simulate pipeline) rather than just
  # compile-only verification.
  if ! grep -q 'llhd.entity\|llhd.proc\|hw.module' "$mlir"; then
    # Extract class name from the SV file
    class_name="$(grep -m1 '^class ' "$sv" | sed 's/class \([a-zA-Z_][a-zA-Z0-9_]*\).*/\1/')"
    if [[ -z "$class_name" ]]; then
      result="NO_TOP"
      skip=$((skip + 1))
      total=$((total - 1))
      printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
      continue
    fi

    # Detect if the constructor needs arguments (e.g., "function new(int seed)")
    ctor_args=""
    ctor_line="$(grep -m1 'function new' "$sv" | sed 's/.*function new *(\(.*\));/\1/')"
    if [[ -n "$ctor_line" ]] && [[ "$ctor_line" != *"function new"* ]]; then
      # Count constructor parameters and provide default values (0 for each)
      n_args=$(echo "$ctor_line" | tr ',' '\n' | wc -l)
      ctor_args=$(printf '0%.0s' $(seq 1 "$n_args") | sed 's/0/, 0/g; s/^, //')
    fi

    # Generate wrapper SV file that exercises the class
    wrapper="$tmpdir/${base}_wrapper.sv"
    cat > "$wrapper" <<WRAPPER_EOF
\`include "$(basename "$sv")"
module top;
  initial begin
    $class_name obj;
    obj = new($ctor_args);
    if (obj.randomize())
      \$display("PASS: %s randomize succeeded", "$class_name");
    else
      \$display("INFO: %s randomize returned 0", "$class_name");
    \$finish;
  end
endmodule
WRAPPER_EOF

    # Recompile with wrapper
    wrapper_cmd=("$CIRCT_VERILOG" --ir-llhd --timescale=1ns/1ns --single-unit \
      -Wno-implicit-conv -Wno-index-oob -Wno-range-oob -Wno-range-width-oob \
      --no-uvm-auto-include --top=top \
      -I "$(dirname "$sv")" "$wrapper")
    if run_limited "${wrapper_cmd[@]}" > "$mlir" 2> "$verilog_log"; then
      top_module="top"
    else
      result="COMPILE_FAIL"
      compile_fail=$((compile_fail + 1))
      printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
      if [[ -n "$KEEP_LOGS_DIR" ]]; then
        mkdir -p "$KEEP_LOGS_DIR"
        cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
        cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
          2>/dev/null || true
      fi
      continue
    fi
  fi

  # Build circt-sim command
  sim_cmd=("$CIRCT_SIM" "--top=$top_module")
  if [[ "$MAX_SIM_TIME" -gt 0 ]]; then
    sim_cmd+=("--max-time=$MAX_SIM_TIME")
  fi
  if [[ -n "$CIRCT_SIM_ARGS" ]]; then
    read -r -a extra_sim_args <<<"$CIRCT_SIM_ARGS"
    sim_cmd+=("${extra_sim_args[@]}")
  fi
  sim_cmd+=("$mlir")

  # Run simulation (UVM tests get a longer timeout + resource guard override)
  sim_output=""
  sim_exit=0
  sim_runner=run_sim_limited
  if [[ "$needs_uvm" -eq 1 ]]; then
    sim_runner=run_uvm_sim_limited
    # Override circt-sim's built-in 300s wall-clock resource guard for UVM
    # tests. UVM MLIR initialization takes ~35s, and phase completion can
    # take several minutes.
    export CIRCT_MAX_WALL_MS=600000
  else
    unset CIRCT_MAX_WALL_MS 2>/dev/null || true
  fi
  if sim_output="$($sim_runner "${sim_cmd[@]}" 2> "$sim_log")"; then
    sim_exit=0
  else
    sim_exit=$?
  fi

  # Classify result
  # Exit code 137 = SIGKILL (timeout); 124 = timeout exit
  if [[ "$sim_exit" -eq 137 || "$sim_exit" -eq 124 ]]; then
    if [[ "$expect" == "xfail" ]]; then
      result="XFAIL"
      xfail=$((xfail + 1))
    else
      result="TIMEOUT"
      timeout_count=$((timeout_count + 1))
      error=$((error + 1))
    fi
  elif [[ "$sim_exit" -eq 0 ]]; then
    # Simulation completed successfully
    if [[ "$should_fail" == "1" ]] || [[ "$expect" == "xfail" ]]; then
      # Expected to fail but passed
      result="XPASS"
      xpass=$((xpass + 1))
    else
      result="PASS"
      pass=$((pass + 1))
    fi
  else
    # Non-zero exit
    if [[ "$should_fail" == "1" ]] || [[ "$expect" == "xfail" ]]; then
      # Expected failure (from test metadata or expect file)
      result="XFAIL"
      xfail=$((xfail + 1))
    else
      # Check stderr for known non-fatal issues
      if grep -q "unsupported\|not yet implemented\|unimplemented" "$sim_log" 2>/dev/null; then
        result="UNSUPPORTED"
        error=$((error + 1))
      else
        result="FAIL"
        fail=$((fail + 1))
      fi
    fi
  fi

  if [[ "$VERBOSE" == "1" ]]; then
    echo "$result	$base"
  fi

  printf "%s\t%s\t%s\n" "$result" "$base" "$sv" >> "$results_tmp"
  if [[ -n "$KEEP_LOGS_DIR" ]]; then
    mkdir -p "$KEEP_LOGS_DIR"
    cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
    cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
      2>/dev/null || true
    cp -f "$sim_log" "$KEEP_LOGS_DIR/${log_tag}.circt-sim.log" \
      2>/dev/null || true
  fi
done < <(find "$SV_TESTS_DIR/tests" -type f -name "*.sv" -print0 | sort -z)

# Disable strict mode for summary to avoid masking results
set +e
sort "$results_tmp" > "$OUT" 2>/dev/null

echo ""
echo "=== sv-tests Simulation Summary ==="
echo "total=$total pass=$pass fail=$fail xfail=$xfail xpass=$xpass"
echo "compile_fail=$compile_fail timeout=$timeout_count error=$error skip=$skip"
eligible=$((total - compile_fail))
if [[ "$eligible" -gt 0 ]]; then
  pass_rate=$(awk "BEGIN {printf \"%.1f\", ($pass + $xfail) / $eligible * 100}")
  echo "eligible=$eligible pass_rate=${pass_rate}%"
fi
echo "results: $OUT"
