#!/bin/bash
# Run a command with memory limit to prevent system hangs
#
# Usage: run_with_memory_limit.sh [OPTIONS] -- command [args...]
#
# Options:
#   -m, --memory LIMIT   Memory limit in GB (default: 20)
#   -t, --timeout SECS   Timeout in seconds (default: 600)
#   -v, --verbose        Print memory limit info
#
# Examples:
#   run_with_memory_limit.sh -- circt-verilog input.sv --ir-hw -o output.mlir
#   run_with_memory_limit.sh -m 10 -t 300 -- circt-sim test.mlir --max-time=1000000

set -e

# Default values
MEMORY_LIMIT_GB=20
TIMEOUT_SECS=600
VERBOSE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--memory)
            MEMORY_LIMIT_GB="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT_SECS="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Error: No command specified" >&2
    echo "Usage: $0 [OPTIONS] -- command [args...]" >&2
    exit 1
fi

# Convert GB to KB for ulimit -v
MEMORY_LIMIT_KB=$((MEMORY_LIMIT_GB * 1024 * 1024))

if [[ $VERBOSE -eq 1 ]]; then
    echo "[run_with_memory_limit] Memory limit: ${MEMORY_LIMIT_GB}GB, Timeout: ${TIMEOUT_SECS}s" >&2
    echo "[run_with_memory_limit] Command: $@" >&2
fi

# Run with memory limit and timeout
# Use a subshell with ulimit to set the memory limit
(
    ulimit -v $MEMORY_LIMIT_KB 2>/dev/null || true
    exec timeout --signal=KILL $TIMEOUT_SECS "$@"
)
exit_code=$?

# Translate exit codes
if [[ $exit_code -eq 137 ]]; then
    echo "[run_with_memory_limit] Process killed (likely OOM or timeout)" >&2
elif [[ $exit_code -eq 124 ]]; then
    echo "[run_with_memory_limit] Process timed out after ${TIMEOUT_SECS}s" >&2
fi

exit $exit_code
