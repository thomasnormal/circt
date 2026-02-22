#!/bin/bash
# Cocotb VPI test runner for circt-sim
# Compiles SV designs to MLIR via circt-verilog, then runs with cocotb via circt-sim --vpi
#
# Usage: bash utils/run_cocotb_tests.sh [test_name ...]
#   No args = run all tests
#   With args = run only named tests (e.g., "bash utils/run_cocotb_tests.sh test_discovery test_cocotb")

set -euo pipefail

# --- Configuration ---
CIRCT_VERILOG="${CIRCT_VERILOG:-build-test/bin/circt-verilog}"
CIRCT_SIM="${CIRCT_SIM:-build-test/bin/circt-sim}"
COCOTB_ROOT="${COCOTB_ROOT:-$HOME/cocotb}"
COCOTB_PKG="$HOME/.local/lib/python3.9/site-packages/cocotb"
VPI_LIB="${VPI_LIB:-$COCOTB_PKG/libs/libcocotbvpi_ius.so}"
WORKDIR="${COCOTB_WORKDIR:-/tmp/cocotb_test}"
MAX_SIM_TIME="${COCOTB_MAX_SIM_TIME:-1000000000000}"  # 1ms in ps (timescale 1ps)

PYGPI_PYTHON_BIN="${PYGPI_PYTHON_BIN:-$(python3 -m cocotb_tools.config --python-bin 2>/dev/null || echo python3)}"
export PYGPI_PYTHON_BIN

DESIGNS="$COCOTB_ROOT/tests/designs"
TESTS="$COCOTB_ROOT/tests/test_cases"

mkdir -p "$WORKDIR/compiled" "$WORKDIR/results"

TOTAL=0
PASS=0
FAIL=0
SKIP=0
FAIL_LIST=""

# --- Compile helper ---
compile_sv() {
    local name="$1"
    shift
    local mlir="$WORKDIR/compiled/${name}.mlir"
    if [ -f "$mlir" ]; then
        return 0
    fi
    echo "  [COMPILE] $name"
    if ! "$CIRCT_VERILOG" --no-uvm-auto-include "$@" -o "$mlir" 2>"$WORKDIR/compiled/${name}.compile.log"; then
        echo "  [COMPILE FAIL] $name — see $WORKDIR/compiled/${name}.compile.log"
        return 1
    fi
    return 0
}

# --- Run helper ---
run_test() {
    local test_name="$1"
    local mlir="$2"
    local top_module="$3"
    local python_module="$4"
    local pythonpath="$5"
    local max_time="${6:-$MAX_SIM_TIME}"
    # Extra env vars passed as "KEY=VAL KEY2=VAL2"
    local extra_env="${7:-}"

    TOTAL=$((TOTAL + 1))

    local result_dir="$WORKDIR/results/$test_name"
    mkdir -p "$result_dir"
    local logfile="$result_dir/sim.log"

    echo -n "  [$TOTAL] $test_name ... "

    # Run simulation in a subshell with exported env vars
    if (
        export COCOTB_TEST_MODULES="$python_module"
        export COCOTB_TOPLEVEL="$top_module"
        export TOPLEVEL_LANG="verilog"
        export PYTHONPATH="$pythonpath"
        export COCOTB_RESULTS_FILE="$result_dir/results.xml"
        # Parse extra env vars
        if [ -n "$extra_env" ]; then
            for kv in $extra_env; do
                export "$kv"
            done
        fi
        timeout 120 "$CIRCT_SIM" "$mlir" --top "$top_module" \
            --vpi "$VPI_LIB" --max-time="$max_time"
    ) >"$logfile" 2>&1; then
        local exit_code=0
    else
        local exit_code=$?
    fi

    # Check for cocotb test failures in results.xml
    if [ -f "$result_dir/results.xml" ]; then
        local failures total_tc
        # Count <failure> elements inside <testcase> elements
        failures=$(grep -c '<failure ' "$result_dir/results.xml" 2>/dev/null) || failures=0
        total_tc=$(grep -c '<testcase ' "$result_dir/results.xml" 2>/dev/null) || total_tc=0
        if [ "$failures" = "0" ]; then
            echo "PASS ($total_tc tests)"
            PASS=$((PASS + 1))
        else
            echo "FAIL ($failures/$total_tc failures, exit=$exit_code)"
            FAIL=$((FAIL + 1))
            FAIL_LIST="$FAIL_LIST $test_name"
        fi
    elif [ "$exit_code" = "0" ]; then
        echo "PASS (no results.xml)"
        PASS=$((PASS + 1))
    else
        echo "FAIL (exit=$exit_code, no results.xml)"
        FAIL=$((FAIL + 1))
        FAIL_LIST="$FAIL_LIST $test_name"
    fi
}

# --- Expected-fail helper (for tests that SHOULD crash/fail) ---
run_test_expected_fail() {
    local test_name="$1"
    local mlir="$2"
    local top_module="$3"
    local python_module="$4"
    local pythonpath="$5"
    local max_time="${6:-$MAX_SIM_TIME}"
    local extra_env="${7:-}"

    TOTAL=$((TOTAL + 1))

    local result_dir="$WORKDIR/results/$test_name"
    mkdir -p "$result_dir"
    local logfile="$result_dir/sim.log"

    local env_cmd="COCOTB_TEST_MODULES=$python_module COCOTB_TOPLEVEL=$top_module TOPLEVEL_LANG=verilog"
    env_cmd="$env_cmd PYTHONPATH=$pythonpath"
    env_cmd="$env_cmd COCOTB_RESULTS_FILE=$result_dir/results.xml"
    [ -n "$extra_env" ] && env_cmd="$env_cmd $extra_env"

    echo -n "  [$TOTAL] $test_name (expected fail) ... "

    if env $env_cmd timeout 120 "$CIRCT_SIM" "$mlir" --top "$top_module" \
        --vpi "$VPI_LIB" --max-time="$max_time" \
        >"$logfile" 2>&1; then
        # Zero exit = unexpected pass
        echo "XPASS (unexpected)"
        FAIL=$((FAIL + 1))
        FAIL_LIST="$FAIL_LIST $test_name"
    else
        echo "XFAIL (expected)"
        PASS=$((PASS + 1))
    fi
}

# --- Check if test is selected ---
should_run() {
    local name="$1"
    if [ ${#SELECTED_TESTS[@]} -eq 0 ]; then
        return 0  # run all
    fi
    for t in "${SELECTED_TESTS[@]}"; do
        if [ "$t" = "$name" ]; then
            return 0
        fi
    done
    return 1
}

# Collect selected tests from command line
SELECTED_TESTS=("$@")

echo "=== Cocotb VPI Test Suite for circt-sim ==="
echo "  circt-verilog: $CIRCT_VERILOG"
echo "  circt-sim:     $CIRCT_SIM"
echo "  VPI library:   $VPI_LIB"
echo "  Work dir:      $WORKDIR"
echo ""

# ============================================================
# Compile designs
# ============================================================
echo "--- Compiling designs ---"

SAMPLE_MODULE_SV="$DESIGNS/sample_module/sample_module.sv"
SAMPLE_MODULE_MLIR="$WORKDIR/compiled/sample_module.mlir"
compile_sv "sample_module" "$SAMPLE_MODULE_SV" -D _VCP || true

BASIC_HIERARCHY_V="$DESIGNS/basic_hierarchy_module/basic_hierarchy_module.v"
BASIC_HIERARCHY_MLIR="$WORKDIR/compiled/basic_hierarchy_module.mlir"
compile_sv "basic_hierarchy_module" "$BASIC_HIERARCHY_V" || true

ARRAY_MODULE_SV="$DESIGNS/array_module/array_module.sv"
ARRAY_MODULE_MLIR="$WORKDIR/compiled/array_module.mlir"
compile_sv "array_module" "$ARRAY_MODULE_SV" || true

MULTI_DIM_PKG="$DESIGNS/multi_dimension_array/cocotb_array_pkg.sv"
MULTI_DIM_SV="$DESIGNS/multi_dimension_array/cocotb_array.sv"
MULTI_DIM_MLIR="$WORKDIR/compiled/multi_dimension_array.mlir"
compile_sv "multi_dimension_array" "$MULTI_DIM_PKG" "$MULTI_DIM_SV" || true

PLUSARGS_V="$DESIGNS/plusargs_module/tb_top.v"
PLUSARGS_MLIR="$WORKDIR/compiled/plusargs_module.mlir"
compile_sv "plusargs_module" "$PLUSARGS_V" || true

# uart2bus - SKIP: mixed-language design (VHDL uart2BusTop module referenced)
# UART2BUS_DIR="$DESIGNS/uart2bus"

# Local SV files from individual tests
for local_test in issue_2255 test_3270 test_defaultless_parameter test_fatal \
    test_first_on_coincident_triggers test_iteration_verilog test_long_log_msg \
    test_package test_packed_union test_verilog_include_dirs; do
    case "$local_test" in
        issue_2255)
            compile_sv "$local_test" "$TESTS/issue_2255/test.sv" || true
            ;;
        test_3270)
            compile_sv "$local_test" "$TESTS/test_3270/test_3270.sv" || true
            ;;
        test_defaultless_parameter)
            compile_sv "$local_test" "$TESTS/test_defaultless_parameter/test_defaultless_parameter.sv" || true
            ;;
        test_fatal)
            compile_sv "$local_test" "$TESTS/test_fatal/fatal.sv" || true
            ;;
        test_first_on_coincident_triggers)
            compile_sv "$local_test" "$TESTS/test_first_on_coincident_triggers/test.sv" || true
            ;;
        test_iteration_verilog)
            compile_sv "$local_test" "$TESTS/test_iteration_verilog/endian_swapper.sv" || true
            ;;
        test_long_log_msg)
            compile_sv "$local_test" "$TESTS/test_long_log_msg/test.sv" || true
            ;;
        test_package)
            compile_sv "$local_test" \
                "$TESTS/test_package/cocotb_package_pkg.sv" \
                "$TESTS/test_package/cocotb_package.sv" || true
            ;;
        test_packed_union)
            compile_sv "$local_test" "$TESTS/test_packed_union/test_packed_union.sv" || true
            ;;
        test_verilog_include_dirs)
            compile_sv "$local_test" "$TESTS/test_verilog_include_dirs/simple_and.sv" \
                -I "$TESTS/test_verilog_include_dirs/common" \
                -I "$TESTS/test_verilog_include_dirs/const_stream" || true
            ;;
    esac
done

echo ""
echo "--- Running tests ---"

# ============================================================
# Group 1: sample_module tests
# ============================================================

SM="$SAMPLE_MODULE_MLIR"

# --- Simple tests (1 run each) ---
for test_name in issue_120 issue_1279 issue_142 issue_348 issue_588 issue_957 \
    test_array_simple test_async_bridge test_discovery test_force_release \
    test_forked_exception test_one_empty_test test_struct; do
    if should_run "$test_name"; then
        run_test "$test_name" "$SM" "sample_module" "$test_name" \
            "$TESTS/$test_name"
    fi
done

# --- test_cocotb (17 Python modules, needs COCOTB_HDL_TIMEPRECISION) ---
if should_run "test_cocotb"; then
    COCOTB_MODULES="test_deprecated,test_synchronization_primitives,test_concurrency_primitives"
    COCOTB_MODULES="$COCOTB_MODULES,test_tests,test_testfactory,test_timing_triggers"
    COCOTB_MODULES="$COCOTB_MODULES,test_scheduler,test_clock,test_edge_triggers"
    COCOTB_MODULES="$COCOTB_MODULES,test_async_coroutines,test_async_generators,test_handle"
    COCOTB_MODULES="$COCOTB_MODULES,test_logging,pytest_assertion_rewriting,test_queues"
    COCOTB_MODULES="$COCOTB_MODULES,test_sim_time_utils,test_start_soon,test_ci"
    run_test "test_cocotb" "$SM" "sample_module" "$COCOTB_MODULES" \
        "$TESTS/test_cocotb" "$MAX_SIM_TIME" \
        "COCOTB_HDL_TIMEPRECISION=1ps"
fi

# --- test_inertial_writes (needs COCOTB_SIMULATOR_TEST=1 for correct expect_fail) ---
if should_run "test_inertial_writes"; then
    run_test "test_inertial_writes" "$SM" "sample_module" "inertial_writes_tests" \
        "$TESTS/test_inertial_writes" "$MAX_SIM_TIME" \
        "COCOTB_SIMULATOR_TEST=1"
fi

# --- test_compare (basic_hierarchy_module) ---
if should_run "test_compare"; then
    run_test "test_compare" "$BASIC_HIERARCHY_MLIR" "basic_hierarchy_module" \
        "test_compare" "$TESTS/test_compare"
fi

# --- test_array (array_module) ---
if should_run "test_array"; then
    run_test "test_array" "$ARRAY_MODULE_MLIR" "array_module" \
        "test_array" "$TESTS/test_array"
fi

# --- test_multi_dimension_array ---
if should_run "test_multi_dimension_array"; then
    run_test "test_multi_dimension_array" "$MULTI_DIM_MLIR" "cocotb_array" \
        "test_cocotb_array" "$TESTS/test_multi_dimension_array"
fi

# --- test_plusargs ---
if should_run "test_plusargs"; then
    TOTAL=$((TOTAL + 1))
    local_result_dir="$WORKDIR/results/test_plusargs"
    mkdir -p "$local_result_dir"
    echo -n "  [$TOTAL] test_plusargs ... "
    if (
        export COCOTB_TEST_MODULES="plusargs"
        export COCOTB_TOPLEVEL="tb_top"
        export TOPLEVEL_LANG="verilog"
        export PYTHONPATH="$TESTS/test_plusargs"
        export COCOTB_RESULTS_FILE="$local_result_dir/results.xml"
        timeout 120 "$CIRCT_SIM" "$PLUSARGS_MLIR" --top "tb_top" \
            --vpi "$VPI_LIB" --max-time="$MAX_SIM_TIME" \
            +foo=bar +test1 +test2 +options=fubar +lol=wow=4
    ) >"$local_result_dir/sim.log" 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi
    if [ -f "$local_result_dir/results.xml" ]; then
        pfail=$(grep -c '<failure ' "$local_result_dir/results.xml" 2>/dev/null) || pfail=0
        ptc=$(grep -c '<testcase ' "$local_result_dir/results.xml" 2>/dev/null) || ptc=0
        if [ "$pfail" = "0" ]; then echo "PASS ($ptc tests)"; PASS=$((PASS + 1))
        else echo "FAIL ($pfail/$ptc failures)"; FAIL=$((FAIL + 1)); FAIL_LIST="$FAIL_LIST test_plusargs"; fi
    elif [ "$exit_code" = "0" ]; then echo "PASS (no results.xml)"; PASS=$((PASS + 1))
    else echo "FAIL (exit=$exit_code)"; FAIL=$((FAIL + 1)); FAIL_LIST="$FAIL_LIST test_plusargs"; fi
fi

# --- test_verilog_access (uart2bus) --- SKIP: mixed-language design

# --- test_seed (2 runs with same seed, check determinism) ---
if should_run "test_seed"; then
    run_test "test_seed_run1" "$SM" "sample_module" "test_other,test_seed" \
        "$TESTS/test_seed" "$MAX_SIM_TIME" \
        "COCOTB_RANDOM_SEED=1234"
    run_test "test_seed_run2" "$SM" "sample_module" "test_other,test_seed" \
        "$TESTS/test_seed" "$MAX_SIM_TIME" \
        "COCOTB_RANDOM_SEED=1234"
fi

# --- test_select_testcase ---
if should_run "test_select_testcase"; then
    run_test "test_select_testcase" "$SM" "sample_module" "x_tests,y_tests,y_tests_again" \
        "$TESTS/test_select_testcase" "$MAX_SIM_TIME" \
        "COCOTB_TESTCASE=y_test"
fi

# --- test_select_testcase_error (deprecated TESTCASE env var, expects no test to match) ---
if should_run "test_select_testcase_error"; then
    run_test "test_select_testcase_error" "$SM" "sample_module" "x_tests" \
        "$TESTS/test_select_testcase_error" "$MAX_SIM_TIME" \
        "TESTCASE=y_test PYTHONWARNINGS=ignore::DeprecationWarning:cocotb._init,"
fi

# --- test_test_filter ---
if should_run "test_test_filter"; then
    run_test "test_test_filter" "$SM" "sample_module" "x_tests" \
        "$TESTS/test_test_filter" "$MAX_SIM_TIME" \
        "COCOTB_TEST_FILTER=y_test"
fi

# --- test_skipped ---
if should_run "test_skipped"; then
    run_test "test_skipped" "$SM" "sample_module" "test_skipped" \
        "$TESTS/test_skipped" "$MAX_SIM_TIME" \
        "COCOTB_TEST_FILTER=test_skipped"
fi

# --- test_multi_level_module_path ---
if should_run "test_multi_level_module_path"; then
    run_test "test_multi_level_module_path" "$SM" "sample_module" \
        "test_package.test_module_path" \
        "$TESTS/test_multi_level_module_path"
fi

# --- Local SV file tests ---
if should_run "issue_2255"; then
    run_test "issue_2255" "$WORKDIR/compiled/issue_2255.mlir" "test" \
        "test_issue2255" "$TESTS/issue_2255" "$MAX_SIM_TIME" \
        "COCOTB_LOG_LEVEL=DEBUG"
fi

if should_run "test_3270"; then
    run_test "test_3270" "$WORKDIR/compiled/test_3270.mlir" "trigger_counter" \
        "test_3270" "$TESTS/test_3270"
fi

if should_run "test_defaultless_parameter"; then
    run_test "test_defaultless_parameter" "$WORKDIR/compiled/test_defaultless_parameter.mlir" \
        "cocotb_defaultless_parameter" "test_defaultless_parameter" \
        "$TESTS/test_defaultless_parameter"
fi

if should_run "test_first_on_coincident_triggers"; then
    run_test "test_first_on_coincident_triggers" \
        "$WORKDIR/compiled/test_first_on_coincident_triggers.mlir" "test" \
        "test_first_on_coincident_triggers" "$TESTS/test_first_on_coincident_triggers"
fi

if should_run "test_iteration_verilog"; then
    run_test "test_iteration_verilog" "$WORKDIR/compiled/test_iteration_verilog.mlir" \
        "endian_swapper_sv" "test_iteration_es" "$TESTS/test_iteration_verilog"
fi

if should_run "test_long_log_msg"; then
    run_test "test_long_log_msg" "$WORKDIR/compiled/test_long_log_msg.mlir" "test" \
        "test_long_log_msg" "$TESTS/test_long_log_msg" "$MAX_SIM_TIME" \
        "COCOTB_LOG_LEVEL=DEBUG"
fi

if should_run "test_package"; then
    run_test "test_package" "$WORKDIR/compiled/test_package.mlir" "cocotb_package" \
        "test_package" "$TESTS/test_package"
fi

if should_run "test_packed_union"; then
    run_test "test_packed_union" "$WORKDIR/compiled/test_packed_union.mlir" \
        "test_packed_union" "test_packed_union" "$TESTS/test_packed_union"
fi

if should_run "test_verilog_include_dirs"; then
    run_test "test_verilog_include_dirs" "$WORKDIR/compiled/test_verilog_include_dirs.mlir" \
        "simple_and" "test_verilog_include_dirs" "$TESTS/test_verilog_include_dirs"
fi

# --- test_fatal ---
if should_run "test_fatal"; then
    run_test "test_fatal" "$WORKDIR/compiled/test_fatal.mlir" "fatal" \
        "test_fatal" "$TESTS/test_fatal"
fi

# --- test_kill_sim (6 sub-runs with expected errors handled by cocotb) ---
if should_run "test_kill_sim"; then
    for filter in test_sys_exit test_task_sys_exit test_trigger_sys_exit \
        test_keyboard_interrupt test_task_keyboard_interrupt test_trigger_keyboard_interrupt; do
        run_test "test_kill_sim_$filter" "$SM" "sample_module" \
            "kill_sim_tests" "$TESTS/test_kill_sim" "$MAX_SIM_TIME" \
            "COCOTB_TEST_FILTER=$filter"
    done
fi

# --- issue_253 (3 sub-runs with different TOPLEVEL settings) ---
if should_run "issue_253"; then
    run_test "issue_253_normal" "$SM" "sample_module" "issue_253" \
        "$TESTS/issue_253" "$MAX_SIM_TIME" \
        "COCOTB_TEST_FILTER=issue_253_none"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "=== Results ==="
echo "  Total: $TOTAL"
echo "  Pass:  $PASS"
echo "  Fail:  $FAIL"
echo "  Skip:  $SKIP"
if [ -n "$FAIL_LIST" ]; then
    echo "  Failed tests:$FAIL_LIST"
fi
echo ""

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
