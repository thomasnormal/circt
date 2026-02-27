#!/usr/bin/env bash
# Deterministic CIRCT AVIP matrix runner.
#
# Produces a per-(avip,seed) matrix at OUT_DIR/matrix.tsv with normalized
# metrics suitable for parity comparisons against Xcelium baselines.
#
# Usage:
#   utils/run_avip_circt_sim.sh [out_dir]
#
# Key env vars:
#   AVIP_SET=core8|all9        (default: core8)
#   AVIPS=comma,list           (overrides AVIP_SET selection)
#   SEEDS=1,2,3                (default: 1)
#   MBIT_DIR=/home/.../mbit
#   COMPILE_TIMEOUT=240        seconds
#   SIM_TIMEOUT=240            seconds
#   SIM_TIMEOUT_GRACE=30       extra seconds before external hard-kill
#   MEMORY_LIMIT_GB=20
#   MAX_WALL_MS=240000
#   CIRCT_SIM_MODE=interpret|compile   (default: interpret)
#   CIRCT_SIM_EXTRA_ARGS="..."         (default: empty)
#   CIRCT_SIM_WRITE_JIT_REPORT=0|1     (default: 0)
#   CIRCT_SIM_DUMP_VCD=0|1             (default: 0)
#   CIRCT_SIM_TRACE_ALL=0|1            (default: 0, auto=1 when dumping VCD)
#   CIRCT_SIM_TRACE_SIGNALS=s0,s1      (optional, repeated --trace)
#   AVIP_SEQUENCE_REPEAT_CAP=N         (default: 8 in interpret mode, 0 otherwise)
#   AVIP_NATIVE_SOURCES_ONLY=0|1       (default: 0)
#   AVIP_SKIP_POST_MOORE_TO_CORE_CLEANUP=0|1 (default: 0)
#   FAIL_ON_ACTIVITY_LIVENESS=0|1       (default: 0)
#   ACTIVITY_MIN_COVERAGE_PCT=0.01      (default: 0.01)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIRCT_ROOT="${CIRCT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUT_DIR="${1:-/tmp/avip-circt-sim-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"

CIRCT_ALLOW_NONCANONICAL_TOOLS="${CIRCT_ALLOW_NONCANONICAL_TOOLS:-0}"

# Optional hard gates (exit non-zero) for CI and unified runners.
FAIL_ON_FUNCTIONAL_GATE="${FAIL_ON_FUNCTIONAL_GATE:-0}"
FAIL_ON_COVERAGE_BASELINE="${FAIL_ON_COVERAGE_BASELINE:-0}"
COVERAGE_BASELINE_PCT="${COVERAGE_BASELINE_PCT:-10}"
FAIL_ON_ACTIVITY_LIVENESS="${FAIL_ON_ACTIVITY_LIVENESS:-0}"
ACTIVITY_MIN_COVERAGE_PCT="${ACTIVITY_MIN_COVERAGE_PCT:-0.01}"

CANONICAL_CIRCT_VERILOG="$CIRCT_ROOT/build_test/bin/circt-verilog"
CANONICAL_CIRCT_SIM="$CIRCT_ROOT/build_test/bin/circt-sim"

CIRCT_VERILOG="${CIRCT_VERILOG:-$CANONICAL_CIRCT_VERILOG}"
CIRCT_SIM="${CIRCT_SIM:-$CANONICAL_CIRCT_SIM}"
RUN_AVIP="${RUN_AVIP:-$SCRIPT_DIR/run_avip_circt_verilog.sh}"
MBIT_DIR="${MBIT_DIR:-/home/thomas-ahle/mbit}"

AVIP_SET="${AVIP_SET:-core8}"
SEEDS_CSV="${SEEDS:-1}"
UVM_VERBOSITY="${UVM_VERBOSITY:-UVM_LOW}"
CIRCT_SIM_MODE="${CIRCT_SIM_MODE:-interpret}"
# AVIPs run UVM-heavy testbenches. Parallel simulation can introduce real host
# thread-level concurrency and break UVM's implicit single-thread assumptions,
# leading to spurious UVM_ERRORs (e.g. uvm_field_op state checks). Also disable
# MLIR internal threading to reduce nondeterminism in complex SV/UVM workloads.
# Callers can override if desired.
CIRCT_SIM_EXTRA_ARGS="${CIRCT_SIM_EXTRA_ARGS:---parallel=1 --mlir-disable-threading}"
CIRCT_SIM_WRITE_JIT_REPORT="${CIRCT_SIM_WRITE_JIT_REPORT:-0}"
CIRCT_SIM_DUMP_VCD="${CIRCT_SIM_DUMP_VCD:-0}"
CIRCT_SIM_TRACE_SIGNALS="${CIRCT_SIM_TRACE_SIGNALS:-}"
if [[ -z "${CIRCT_SIM_TRACE_ALL+x}" ]]; then
  if [[ "$CIRCT_SIM_DUMP_VCD" != "0" ]]; then
    CIRCT_SIM_TRACE_ALL=1
  else
    CIRCT_SIM_TRACE_ALL=0
  fi
fi
# AVIP monitor diagnostic rewrites are useful for bring-up but very noisy in
# interpreted regressions. Keep them opt-in and strip heavy per-cycle display
# spam by default in interpreted mode.
AVIP_ENABLE_MONITOR_DIAG_REWRITES="${AVIP_ENABLE_MONITOR_DIAG_REWRITES:-0}"
AVIP_SKIP_POST_MOORE_TO_CORE_CLEANUP="${AVIP_SKIP_POST_MOORE_TO_CORE_CLEANUP:-0}"
VERILOG_EXTRA_ARGS="${CIRCT_VERILOG_ARGS:-}"
if [[ "$AVIP_SKIP_POST_MOORE_TO_CORE_CLEANUP" == "1" ]]; then
  if [[ " $VERILOG_EXTRA_ARGS " != *" --skip-post-moore-to-core-cleanup "* ]]; then
    if [[ -n "$VERILOG_EXTRA_ARGS" ]]; then
      VERILOG_EXTRA_ARGS+=" "
    fi
    VERILOG_EXTRA_ARGS+="--skip-post-moore-to-core-cleanup"
  fi
fi
if [[ -z "${AVIP_STRIP_MONITOR_DISPLAYS+x}" ]]; then
  if [[ "$CIRCT_SIM_MODE" == "interpret" ]]; then
    AVIP_STRIP_MONITOR_DISPLAYS=1
  else
    AVIP_STRIP_MONITOR_DISPLAYS=0
  fi
fi
if [[ -z "${AVIP_SEQUENCE_REPEAT_CAP+x}" ]]; then
  if [[ "$CIRCT_SIM_MODE" == "interpret" ]]; then
    # Interpreted mode can be significantly slower on full-length UVM loops.
    # Cap fixed repeat() counts in sequence/test sources via the verilog runner
    # rewrite path to preserve e2e flow while keeping smoke runs bounded.
    AVIP_SEQUENCE_REPEAT_CAP=8
  else
    AVIP_SEQUENCE_REPEAT_CAP=0
  fi
fi
AVIP_NATIVE_SOURCES_ONLY="${AVIP_NATIVE_SOURCES_ONLY:-0}"
if [[ -z "${AVIP_RELAX_RESET_WAIT+x}" ]]; then
  if [[ "$CIRCT_SIM_MODE" == "interpret" ]]; then
    AVIP_RELAX_RESET_WAIT=1
  else
    AVIP_RELAX_RESET_WAIT=0
  fi
fi

MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-20}"
COMPILE_TIMEOUT="${COMPILE_TIMEOUT:-240}"
SIM_TIMEOUT="${SIM_TIMEOUT:-240}"
SIM_TIMEOUT_GRACE="${SIM_TIMEOUT_GRACE:-30}"
SIM_RETRIES="${SIM_RETRIES:-0}"
SIM_RETRY_ON_FCTTYP="${SIM_RETRY_ON_FCTTYP:-1}"
SIM_RETRY_ON_CRASH="${SIM_RETRY_ON_CRASH:-1}"
SIM_RETRY_ON_UVM_FIELD_OP="${SIM_RETRY_ON_UVM_FIELD_OP:-1}"
SIM_RETRY_ON_TIMEOUT="${SIM_RETRY_ON_TIMEOUT:-1}"
SIM_TIMEOUT_HARD=$((SIM_TIMEOUT + SIM_TIMEOUT_GRACE))
if [[ -z "${MAX_WALL_MS+x}" ]]; then
  MAX_WALL_MS="$((SIM_TIMEOUT_HARD * 1000))"
fi
MEMORY_LIMIT_KB=$((MEMORY_LIMIT_GB * 1024 * 1024))

TIME_TOOL=""
if [[ -x /usr/bin/time ]]; then
  TIME_TOOL="/usr/bin/time"
elif command -v gtime >/dev/null 2>&1; then
  TIME_TOOL="$(command -v gtime)"
fi

if [[ "$CIRCT_ALLOW_NONCANONICAL_TOOLS" != "1" ]]; then
  if [[ -n "${CIRCT_VERILOG:-}" && "$CIRCT_VERILOG" != "$CANONICAL_CIRCT_VERILOG" ]]; then
    echo "error: CIRCT_VERILOG must use canonical build_test path: $CANONICAL_CIRCT_VERILOG (got: $CIRCT_VERILOG)" >&2
    exit 1
  fi
  if [[ -n "${CIRCT_SIM:-}" && "$CIRCT_SIM" != "$CANONICAL_CIRCT_SIM" ]]; then
    echo "error: CIRCT_SIM must use canonical build_test path: $CANONICAL_CIRCT_SIM (got: $CIRCT_SIM)" >&2
    exit 1
  fi
fi

tool_help_healthy() {
  local tool="$1"
  [[ -x "$tool" ]] || return 1
  "$tool" --help >/dev/null 2>&1
}

if [[ "$CIRCT_ALLOW_NONCANONICAL_TOOLS" != "1" ]]; then
  if [[ -x "$CIRCT_SIM" ]] && ! tool_help_healthy "$CIRCT_SIM"; then
    echo "error: circt-sim probe failed for '$CIRCT_SIM' (rebuild that binary)" >&2
    exit 1
  fi
fi

if [[ ! -x "$CIRCT_SIM" ]]; then
  echo "error: circt-sim not found or not executable: $CIRCT_SIM" >&2
  exit 1
fi
if [[ ! -x "$CIRCT_VERILOG" ]]; then
  echo "error: circt-verilog not found or not executable: $CIRCT_VERILOG" >&2
  exit 1
fi
if [[ ! -x "$RUN_AVIP" ]]; then
  echo "error: helper runner not found or not executable: $RUN_AVIP" >&2
  exit 1
fi

# Snapshot tool binaries once per run to avoid races with concurrent rebuilds.
TOOL_SNAPSHOT_DIR="$OUT_DIR/.tool-snapshot"
mkdir -p "$TOOL_SNAPSHOT_DIR"

snapshot_tool() {
  local src="$1"
  local dst="$2"
  if ! cp -f "$src" "$dst"; then
    echo "error: failed to snapshot tool: $src -> $dst" >&2
    exit 1
  fi
  chmod +x "$dst" 2>/dev/null || true
}

SNAPSHOT_CIRCT_VERILOG="$TOOL_SNAPSHOT_DIR/circt-verilog"
SNAPSHOT_CIRCT_SIM="$TOOL_SNAPSHOT_DIR/circt-sim"
snapshot_tool "$CIRCT_VERILOG" "$SNAPSHOT_CIRCT_VERILOG"
snapshot_tool "$CIRCT_SIM" "$SNAPSHOT_CIRCT_SIM"

# Use stable snapshots for the entire matrix execution.
CIRCT_VERILOG="$SNAPSHOT_CIRCT_VERILOG"
CIRCT_SIM="$SNAPSHOT_CIRCT_SIM"

# name|avip_dir|filelist|tops|max_sim_fs|test_name
AVIPS_CORE8=(
  "apb|$MBIT_DIR/apb_avip|$MBIT_DIR/apb_avip/sim/apb_compile.f|hdl_top,hvl_top|2260000000000|apb_8b_write_test"
  "ahb|$MBIT_DIR/ahb_avip|$MBIT_DIR/ahb_avip/sim/ahb_compile.f|HdlTop,HvlTop|20620000000000|AhbWriteTest"
  "axi4|$MBIT_DIR/axi4_avip|$MBIT_DIR/axi4_avip/sim/axi4_compile.f|hdl_top,hvl_top|9060000000000|axi4_write_read_test"
  # Use a test that is included in the default Axi4LiteProject.f filelist.
  # (The MasterVIPSlaveIPWriteTest lives under examples/ and is not compiled by default.)
  "axi4Lite|$MBIT_DIR/axi4Lite_avip|$MBIT_DIR/axi4Lite_avip/sim/Axi4LiteProject.f|Axi4LiteHdlTop,Axi4LiteHvlTop|10000000000000|Axi4LiteWriteTest"
  "i2s|$MBIT_DIR/i2s_avip|$MBIT_DIR/i2s_avip/sim/I2sCompile.f|hdlTop,hvlTop|84840000000000|I2sWriteOperationWith8bitdataTxMasterRxSlaveWith48khzTest"
  "i3c|$MBIT_DIR/i3c_avip|$MBIT_DIR/i3c_avip/sim/i3c_compile.f|hdl_top,hvl_top|7940000000000|i3c_writeOperationWith8bitsData_test"
  "jtag|$MBIT_DIR/jtag_avip|$MBIT_DIR/jtag_avip/sim/JtagCompile.f|HdlTop,HvlTop|369400000000|JtagTdiWidth24Test"
  "spi|$MBIT_DIR/spi_avip|$MBIT_DIR/spi_avip/sim/SpiCompile.f|SpiHdlTop,SpiHvlTop|4420000000000|SpiSimpleFd8BitsTest"
)
AVIPS_ALL9=(
  "apb|$MBIT_DIR/apb_avip|$MBIT_DIR/apb_avip/sim/apb_compile.f|hdl_top,hvl_top|2260000000000|apb_8b_write_test"
  "ahb|$MBIT_DIR/ahb_avip|$MBIT_DIR/ahb_avip/sim/ahb_compile.f|HdlTop,HvlTop|20620000000000|AhbWriteTest"
  "axi4|$MBIT_DIR/axi4_avip|$MBIT_DIR/axi4_avip/sim/axi4_compile.f|hdl_top,hvl_top|9060000000000|axi4_write_read_test"
  # Use a test that is included in the default Axi4LiteProject.f filelist.
  # (The MasterVIPSlaveIPWriteTest lives under examples/ and is not compiled by default.)
  "axi4Lite|$MBIT_DIR/axi4Lite_avip|$MBIT_DIR/axi4Lite_avip/sim/Axi4LiteProject.f|Axi4LiteHdlTop,Axi4LiteHvlTop|10000000000000|Axi4LiteWriteTest"
  "i2s|$MBIT_DIR/i2s_avip|$MBIT_DIR/i2s_avip/sim/I2sCompile.f|hdlTop,hvlTop|84840000000000|I2sWriteOperationWith8bitdataTxMasterRxSlaveWith48khzTest"
  "i3c|$MBIT_DIR/i3c_avip|$MBIT_DIR/i3c_avip/sim/i3c_compile.f|hdl_top,hvl_top|7940000000000|i3c_writeOperationWith8bitsData_test"
  "jtag|$MBIT_DIR/jtag_avip|$MBIT_DIR/jtag_avip/sim/JtagCompile.f|HdlTop,HvlTop|369400000000|JtagTdiWidth24Test"
  "spi|$MBIT_DIR/spi_avip|$MBIT_DIR/spi_avip/sim/SpiCompile.f|SpiHdlTop,SpiHvlTop|4420000000000|SpiSimpleFd8BitsTest"
  "uart|$MBIT_DIR/uart_avip|$MBIT_DIR/uart_avip/sim/UartCompile.f|HdlTop,HvlTop|10000000000000|UartBaudRate4800Test"
)

selected_avips=()
if [[ -n "${AVIPS:-}" ]]; then
  IFS=',' read -ra requested <<< "$AVIPS"
  for want in "${requested[@]}"; do
    want="${want//[[:space:]]/}"
    [[ -z "$want" ]] && continue
    found=0
    for row in "${AVIPS_ALL9[@]}"; do
      IFS='|' read -r name _ <<< "$row"
      if [[ "$name" == "$want" ]]; then
        selected_avips+=("$row")
        found=1
        break
      fi
    done
    if [[ $found -eq 0 ]]; then
      echo "warning: unknown AVIP '$want' (skipping)" >&2
    fi
  done
else
  case "$AVIP_SET" in
    core8) selected_avips=("${AVIPS_CORE8[@]}") ;;
    all9) selected_avips=("${AVIPS_ALL9[@]}") ;;
    *)
      echo "error: unsupported AVIP_SET '$AVIP_SET' (use core8 or all9)" >&2
      exit 1
      ;;
  esac
fi

if [[ ${#selected_avips[@]} -eq 0 ]]; then
  echo "error: no AVIPs selected" >&2
  exit 1
fi

IFS=',' read -ra seeds <<< "$SEEDS_CSV"
if [[ ${#seeds[@]} -eq 0 ]]; then
  echo "error: no seeds selected" >&2
  exit 1
fi

case "$CIRCT_SIM_MODE" in
  interpret|compile) ;;
  *)
    echo "error: unsupported CIRCT_SIM_MODE '$CIRCT_SIM_MODE' (use interpret or compile)" >&2
    exit 1
    ;;
esac

if [[ ! "$SIM_RETRIES" =~ ^[0-9]+$ ]]; then
  echo "error: SIM_RETRIES must be a non-negative integer (got '$SIM_RETRIES')" >&2
  exit 1
fi

if ! [[ "$ACTIVITY_MIN_COVERAGE_PCT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "error: ACTIVITY_MIN_COVERAGE_PCT must be a non-negative number (got '$ACTIVITY_MIN_COVERAGE_PCT')" >&2
  exit 1
fi
if [[ "$CIRCT_SIM_DUMP_VCD" != "0" && "$CIRCT_SIM_DUMP_VCD" != "1" ]]; then
  echo "error: CIRCT_SIM_DUMP_VCD must be 0 or 1 (got '$CIRCT_SIM_DUMP_VCD')" >&2
  exit 1
fi
if [[ "$CIRCT_SIM_TRACE_ALL" != "0" && "$CIRCT_SIM_TRACE_ALL" != "1" ]]; then
  echo "error: CIRCT_SIM_TRACE_ALL must be 0 or 1 (got '$CIRCT_SIM_TRACE_ALL')" >&2
  exit 1
fi
if ! [[ "$AVIP_SEQUENCE_REPEAT_CAP" =~ ^[0-9]+$ ]]; then
  echo "error: AVIP_SEQUENCE_REPEAT_CAP must be a non-negative integer (got '$AVIP_SEQUENCE_REPEAT_CAP')" >&2
  exit 1
fi

sim_extra_args=()
if [[ -n "$CIRCT_SIM_EXTRA_ARGS" ]]; then
  # Parsed as shell words; quote-preserving parsing is intentionally not used.
  read -r -a sim_extra_args <<< "$CIRCT_SIM_EXTRA_ARGS"
fi
trace_flags=()
if [[ "$CIRCT_SIM_TRACE_ALL" == "1" ]]; then
  trace_flags+=(--trace-all)
fi
if [[ -n "$CIRCT_SIM_TRACE_SIGNALS" ]]; then
  IFS=',' read -r -a trace_signals <<< "$CIRCT_SIM_TRACE_SIGNALS"
  for trace_sig in "${trace_signals[@]}"; do
    trace_sig="${trace_sig//[[:space:]]/}"
    [[ -z "$trace_sig" ]] && continue
    trace_flags+=(--trace "$trace_sig")
  done
fi

run_limited() {
  local timeout_secs="$1"
  shift
  (
    ulimit -v "$MEMORY_LIMIT_KB" 2>/dev/null || true
    timeout --signal=KILL "$timeout_secs" "$@"
  )
}

resolve_compile_filelist() {
  local avip_dir="$1"
  local preferred="$2"
  if [[ -n "$preferred" && -f "$preferred" ]]; then
    printf '%s\n' "$preferred"
    return 0
  fi
  if [[ -d "$avip_dir/sim" ]]; then
    local candidate=""
    candidate="$(find "$avip_dir/sim" -maxdepth 3 -type f \( -iname "*compile*.f" -o -iname "*project*.f" \) | head -n 1 || true)"
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  printf '%s\n' ""
}

extract_uvm_count() {
  local key="$1"
  local log="$2"
  local val
  # Prefer explicit UVM summary lines if present (some harnesses print them).
  val=$(grep -Eo "${key}[[:space:]]*:[[:space:]]*[0-9]+" "$log" | tail -1 | sed -E 's/.*:[[:space:]]*([0-9]+)/\1/' || true)
  if [[ -n "$val" ]]; then
    echo "$val"
    return 0
  fi

  # Secondary parse path: count individual occurrences. Return 0 if none; '?'
  # is reserved
  # for truly unknown/parse failures.
  local count
  count=$(grep -Ec "^${key}([[:space:]]|$)" "$log" 2>/dev/null || true)
  if [[ -n "$count" ]]; then
    echo "$count"
  else
    echo "?"
  fi
}

extract_sim_time_fs() {
  local log="$1"
  local val
  # circt-sim currently prints "[circt-sim] Simulation completed at time ..."
  # but some older snapshots used "Simulation terminated at time ...".
  val=$(
    grep -Eo 'Simulation (completed|terminated) at time[[:space:]]+[0-9]+[[:space:]]+fs' "$log" \
      | tail -1 \
      | sed -E 's/.*time[[:space:]]+([0-9]+)[[:space:]]+fs/\1/' \
      || true
  )
  [[ -n "$val" ]] && echo "$val" || echo "-"
}

extract_coverage_pair() {
  local log="$1"
  local values=""
  # circt-sim prints a coverage report with "Overall coverage: <pct>%".
  values=$(
    grep -Eo 'Overall coverage:[[:space:]]*[0-9]+(\.[0-9]+)?%' "$log" \
      | sed -E 's/.*Overall coverage:[[:space:]]*([0-9]+(\.[0-9]+)?)%/\1/' \
      | tr '\n' ' ' \
      || true
  )
  # Back-compat: older harnesses used "Coverage = <pct> %".
  if [[ -z "${values//[[:space:]]/}" ]]; then
    values=$(
      grep -Eo 'Coverage[[:space:]]*=[[:space:]]*[0-9]+(\.[0-9]+)?[[:space:]]*%' "$log" \
        | sed -E 's/.*=[[:space:]]*([0-9]+(\.[0-9]+)?)[[:space:]]*%/\1/' \
        | tr '\n' ' ' \
        || true
    )
  fi
  local cov1="-"
  local cov2="-"
  if [[ -n "$values" ]]; then
    read -r cov1 cov2 _ <<< "$values"
    [[ -z "$cov1" ]] && cov1="-"
    [[ -z "$cov2" ]] && cov2="-"
  fi
  echo "$cov1|$cov2"
}

log_has_wall_timeout() {
  local log="$1"
  grep -q "Wall-clock timeout reached (global guard)" "$log" 2>/dev/null
}

log_has_sim_completed() {
  local log="$1"
  grep -Eq "Simulation (completed|terminated) at time[[:space:]]+[0-9]+[[:space:]]+fs" "$log" 2>/dev/null
}

log_has_virtual_call_failure() {
  local log="$1"
  grep -Fq "[circt-sim] WARNING: virtual method call" "$log" 2>/dev/null
}

log_has_uvm_field_op_failure() {
  local log="$1"
  grep -Fq "UVM/FIELD_OP/SET Attempting to set values in policy without flushing" "$log" 2>/dev/null && \
    grep -Fq "UVM/FIELD_OP/GET_OP_TYPE Calling get_op_type() before calling set() is not allowed" "$log" 2>/dev/null
}

sim_status_from_exit() {
  local code="$1"
  case "$code" in
    0) echo "OK" ;;
    124|137) echo "TIMEOUT" ;;
    *) echo "FAIL" ;;
  esac
}

should_retry_sim_failure() {
  local code="$1"
  local log="$2"
  if [[ "$SIM_RETRY_ON_TIMEOUT" != "0" ]] && [[ "$code" -eq 124 || "$code" -eq 137 ]]; then
    return 0
  fi
  if [[ "$SIM_RETRY_ON_FCTTYP" != "0" ]] && [[ "$code" -ne 0 ]] && \
     [[ -f "$log" ]] && grep -q "UVM_FATAL @ 0: FCTTYP" "$log"; then
    return 0
  fi

  if [[ "$SIM_RETRY_ON_UVM_FIELD_OP" != "0" ]] && [[ "$code" -ne 0 ]] && \
     [[ -f "$log" ]] && log_has_uvm_field_op_failure "$log"; then
    return 0
  fi

  # Treat interpreter crashes as transient infra failures and retry if enabled.
  # These show up as a non-zero exit code (often 139) and/or LLVM's crash
  # handler banner in the log. Empirically, rerunning the same MLIR can succeed.
  if [[ "$SIM_RETRY_ON_CRASH" != "0" ]] && [[ "$code" -ne 0 ]] && [[ -f "$log" ]]; then
    if [[ "$code" -eq 139 ]] || \
       grep -q "PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/" "$log" || \
       grep -q "timeout: the monitored command dumped core" "$log"; then
      return 0
    fi
  fi
  return 1
}

matrix="$OUT_DIR/matrix.tsv"
cat > "$matrix" <<'HDR'
avip	seed	compile_status	compile_sec	sim_status	sim_exit	sim_sec	sim_time_fs	uvm_fatal	uvm_error	cov_1_pct	cov_2_pct	peak_rss_kb	compile_log	sim_log	vcd_file
HDR

meta="$OUT_DIR/meta.txt"
{
  echo "generated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "script_path=$0"
  echo "circt_root=$CIRCT_ROOT"
  echo "circt_verilog=$CIRCT_VERILOG"
  echo "circt_sim=$CIRCT_SIM"
  echo "avip_set=$AVIP_SET"
  echo "avips=${AVIPS:-<from-set>}"
  echo "seeds=$SEEDS_CSV"
  echo "memory_limit_gb=$MEMORY_LIMIT_GB"
  echo "compile_timeout=$COMPILE_TIMEOUT"
  echo "sim_timeout=$SIM_TIMEOUT"
  echo "sim_timeout_grace=$SIM_TIMEOUT_GRACE"
  echo "sim_timeout_hard=$SIM_TIMEOUT_HARD"
  echo "max_wall_ms=$MAX_WALL_MS"
  echo "circt_sim_mode=$CIRCT_SIM_MODE"
  echo "circt_sim_extra_args=${CIRCT_SIM_EXTRA_ARGS:-<none>}"
  echo "circt_sim_write_jit_report=$CIRCT_SIM_WRITE_JIT_REPORT"
  echo "circt_sim_dump_vcd=$CIRCT_SIM_DUMP_VCD"
  echo "circt_sim_trace_all=$CIRCT_SIM_TRACE_ALL"
  echo "circt_sim_trace_signals=${CIRCT_SIM_TRACE_SIGNALS:-<none>}"
  echo "circt_verilog_args=${VERILOG_EXTRA_ARGS:-<none>}"
  echo "avip_sequence_repeat_cap=$AVIP_SEQUENCE_REPEAT_CAP"
  echo "avip_native_sources_only=$AVIP_NATIVE_SOURCES_ONLY"
  echo "avip_skip_post_moore_to_core_cleanup=$AVIP_SKIP_POST_MOORE_TO_CORE_CLEANUP"
  echo "avip_relax_reset_wait=$AVIP_RELAX_RESET_WAIT"
  echo "fail_on_functional_gate=$FAIL_ON_FUNCTIONAL_GATE"
  echo "fail_on_coverage_baseline=$FAIL_ON_COVERAGE_BASELINE"
  echo "coverage_baseline_pct=$COVERAGE_BASELINE_PCT"
  echo "fail_on_activity_liveness=$FAIL_ON_ACTIVITY_LIVENESS"
  echo "activity_min_coverage_pct=$ACTIVITY_MIN_COVERAGE_PCT"
  echo "time_tool=${TIME_TOOL:-<none>}"
} > "$meta"

echo "[avip-circt-sim] out_dir=$OUT_DIR"
echo "[avip-circt-sim] selected_avips=${#selected_avips[@]} seeds=${#seeds[@]}"

for row in "${selected_avips[@]}"; do
  IFS='|' read -r name avip_dir filelist tops max_sim_fs test_name <<< "$row"
  avip_out="$OUT_DIR/$name"
  mkdir -p "$avip_out"

  sim_tops="$tops"
  sim_test_name="$test_name"
  compile_log="$avip_out/compile.log"
  mlir_file="$avip_out/${name}.mlir"

  compile_status="FAIL"
  compile_sec="-"

  if [[ ! -d "$avip_dir" ]]; then
    echo "warning: missing AVIP directory: $avip_dir" >&2
  else
    start_compile=$(date +%s)
    resolved_filelist="$(resolve_compile_filelist "$avip_dir" "$filelist")"
    if [[ "$name" == "axi4Lite" ]] && [[ -n "$filelist" ]] && [[ ! -f "$filelist" ]]; then
      # Some local AVIP trees do not ship sim/Axi4LiteProject.f. In that case
      # run_avip_circt_verilog falls back to nested master/slave filelists,
      # which provide the master tops/tests but not the combined Axi4LiteHdlTop.
      sim_tops="${AXI4LITE_FALLBACK_TOPS:-Axi4LiteMasterHdlTop,Axi4LiteMasterHvlTop}"
      sim_test_name="${AXI4LITE_FALLBACK_TESTNAME:-Axi4LiteMasterRandomWriteReadTransferTest}"
      echo "[avip-circt-sim] info: using AXI4Lite fallback tops/test (missing $filelist): tops=$sim_tops test=$sim_test_name"
    fi
    compile_cmd=("$RUN_AVIP" "$avip_dir")
    if [[ -n "$resolved_filelist" ]]; then
      compile_cmd+=("$resolved_filelist")
    else
      echo "[avip-circt-sim] info: no explicit sim filelist for '$name'; using runner auto filelist resolution"
    fi
    if CIRCT_VERILOG="$CIRCT_VERILOG" \
       CIRCT_VERILOG_IR=llhd \
       CIRCT_VERILOG_ARGS="$VERILOG_EXTRA_ARGS" \
       AVIP_ENABLE_MONITOR_DIAG_REWRITES="$AVIP_ENABLE_MONITOR_DIAG_REWRITES" \
       AVIP_STRIP_MONITOR_DISPLAYS="$AVIP_STRIP_MONITOR_DISPLAYS" \
       AVIP_SEQUENCE_REPEAT_CAP="$AVIP_SEQUENCE_REPEAT_CAP" \
       AVIP_NATIVE_SOURCES_ONLY="$AVIP_NATIVE_SOURCES_ONLY" \
       AVIP_RELAX_RESET_WAIT="$AVIP_RELAX_RESET_WAIT" \
       OUT="$mlir_file" \
       run_limited "$COMPILE_TIMEOUT" "${compile_cmd[@]}" > "$compile_log" 2>&1; then
      compile_status="OK"
    fi
    end_compile=$(date +%s)
    compile_sec=$((end_compile - start_compile))
  fi

  echo "[avip-circt-sim] $name compile_status=$compile_status compile_sec=${compile_sec}s"

  for seed in "${seeds[@]}"; do
    seed="${seed//[[:space:]]/}"
    [[ -z "$seed" ]] && continue

    sim_status="SKIP"
    sim_exit="-"
    sim_sec="-"
    sim_time_fs="-"
    uvm_fatal="-"
    uvm_error="-"
    cov_1="-"
    cov_2="-"
    peak_rss_kb="-"
    vcd_file="-"

    sim_log="$avip_out/sim_seed_${seed}.log"
    rss_log="$avip_out/sim_seed_${seed}.rss_kb"
    jit_report="$avip_out/sim_seed_${seed}.jit-report.json"
    vcd_run_file="$avip_out/sim_seed_${seed}.vcd"
    vcd_flags=()
    if [[ "$CIRCT_SIM_DUMP_VCD" == "1" ]]; then
      vcd_file="$vcd_run_file"
      vcd_flags+=(--vcd="$vcd_run_file")
    fi

    if [[ "$compile_status" == "OK" ]]; then
      top_flags=()
      IFS=',' read -ra top_modules <<< "$sim_tops"
      for t in "${top_modules[@]}"; do
        t="${t//[[:space:]]/}"
        [[ -n "$t" ]] && top_flags+=(--top "$t")
      done

      uvm_args="+ntb_random_seed=$seed +UVM_VERBOSITY=$UVM_VERBOSITY"
      if [[ -n "$sim_test_name" ]]; then
        uvm_args="+UVM_TESTNAME=$sim_test_name $uvm_args"
      fi

      mode_flags=(--mode="$CIRCT_SIM_MODE")
      jit_report_flags=()
      if [[ "$CIRCT_SIM_WRITE_JIT_REPORT" != "0" ]]; then
        jit_report_flags+=(--jit-report="$jit_report")
      fi

      start_sim=$(date +%s)
      retry_count=0
      while true; do
        set +e
        if [[ -n "$TIME_TOOL" ]]; then
          run_limited "$SIM_TIMEOUT_HARD" \
            "$TIME_TOOL" -f "%M" -o "$rss_log" \
            env CIRCT_UVM_ARGS="$uvm_args" \
            "$CIRCT_SIM" "$mlir_file" \
            "${top_flags[@]}" \
            "${mode_flags[@]}" \
            "${sim_extra_args[@]}" \
            "${trace_flags[@]}" \
            "${vcd_flags[@]}" \
            "${jit_report_flags[@]}" \
            --timeout="$SIM_TIMEOUT" \
            --max-time="$max_sim_fs" \
            --max-wall-ms="$MAX_WALL_MS" \
            > "$sim_log" 2>&1
        else
          run_limited "$SIM_TIMEOUT_HARD" \
            env CIRCT_UVM_ARGS="$uvm_args" \
            "$CIRCT_SIM" "$mlir_file" \
            "${top_flags[@]}" \
            "${mode_flags[@]}" \
            "${sim_extra_args[@]}" \
            "${trace_flags[@]}" \
            "${vcd_flags[@]}" \
            "${jit_report_flags[@]}" \
            --timeout="$SIM_TIMEOUT" \
            --max-time="$max_sim_fs" \
            --max-wall-ms="$MAX_WALL_MS" \
            > "$sim_log" 2>&1
        fi
        sim_exit=$?
        set -e

        if [[ "$sim_exit" -eq 0 ]]; then
          # Treat internal virtual-dispatch failures as retry-worthy infra
          # errors. These can appear transiently and often disappear on rerun.
          if log_has_virtual_call_failure "$sim_log" && (( retry_count < SIM_RETRIES )); then
            retry_count=$((retry_count + 1))
            cp -f "$sim_log" "$sim_log.attempt${retry_count}.log" 2>/dev/null || true
            continue
          fi
          if [[ "$SIM_RETRY_ON_UVM_FIELD_OP" != "0" ]] && log_has_uvm_field_op_failure "$sim_log" && \
             (( retry_count < SIM_RETRIES )); then
            retry_count=$((retry_count + 1))
            cp -f "$sim_log" "$sim_log.attempt${retry_count}.log" 2>/dev/null || true
            continue
          fi
          break
        fi
        if (( retry_count >= SIM_RETRIES )); then
          break
        fi
        if ! should_retry_sim_failure "$sim_exit" "$sim_log"; then
          break
        fi

        retry_count=$((retry_count + 1))
        cp -f "$sim_log" "$sim_log.attempt${retry_count}.log" 2>/dev/null || true
      done
      end_sim=$(date +%s)
      sim_sec=$((end_sim - start_sim))

      sim_status=$(sim_status_from_exit "$sim_exit")
      sim_time_fs=$(extract_sim_time_fs "$sim_log")
      uvm_fatal=$(extract_uvm_count "UVM_FATAL" "$sim_log")
      uvm_error=$(extract_uvm_count "UVM_ERROR" "$sim_log")
      cov_pair=$(extract_coverage_pair "$sim_log")
      IFS='|' read -r cov_1 cov_2 <<< "$cov_pair"

      # If the simulator printed a global wall timeout marker and did not also
      # print a completed/terminated timestamp, treat this as a timeout even
      # if the process exit code was zero.
      if log_has_wall_timeout "$sim_log" && ! log_has_sim_completed "$sim_log"; then
        sim_status="TIMEOUT"
        sim_time_fs="-"
      fi

      # UVM failures should count as failures regardless of process exit code.
      if [[ "$uvm_fatal" != "0" || "$uvm_error" != "0" ]]; then
        sim_status="FAIL"
      fi

      if [[ -f "$rss_log" ]]; then
        peak_rss_kb=$(tail -n 1 "$rss_log" | tr -d '[:space:]')
        [[ -z "$peak_rss_kb" ]] && peak_rss_kb="-"
      fi
    fi

    if [[ "$vcd_file" != "-" && ! -s "$vcd_file" ]]; then
      vcd_file="-"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$name" "$seed" "$compile_status" "$compile_sec" "$sim_status" "$sim_exit" \
      "$sim_sec" "$sim_time_fs" "$uvm_fatal" "$uvm_error" "$cov_1" "$cov_2" \
      "$peak_rss_kb" "$compile_log" "$sim_log" "$vcd_file" >> "$matrix"

    echo "[avip-circt-sim] $name seed=$seed sim_status=$sim_status sim_exit=$sim_exit sim_sec=${sim_sec}s sim_time_fs=$sim_time_fs rss_kb=$peak_rss_kb vcd=$vcd_file"
  done
done

echo "[avip-circt-sim] matrix=$matrix"
echo "[avip-circt-sim] meta=$meta"

functional_fail_rows="$(
  awk -F'\t' '
    NR == 1 { next }
    {
      compile = $3
      sim_status = $5
      sim_exit = $6
      uvm_fatal = $9
      uvm_error = $10
      bad = 0
      if (compile != "OK") bad = 1
      else if (sim_status != "OK") bad = 1
      else if (sim_exit != "0") bad = 1
      else if (uvm_fatal == "-" || uvm_fatal == "?" || uvm_fatal == "") bad = 1
      else if (uvm_error == "-" || uvm_error == "?" || uvm_error == "") bad = 1
      else if (uvm_fatal + 0 != 0) bad = 1
      else if (uvm_error + 0 != 0) bad = 1
      if (bad) n += 1
    }
    END { print n + 0 }
  ' "$matrix" || true
)"

coverage_fail_rows="$(
  awk -F'\t' -v base="$COVERAGE_BASELINE_PCT" '
    NR == 1 { next }
    {
      compile = $3
      sim_status = $5
      sim_exit = $6
      uvm_fatal = $9
      uvm_error = $10
      cov1 = $11
      cov2 = $12

      functional_pass = 1
      if (compile != "OK") functional_pass = 0
      else if (sim_status != "OK") functional_pass = 0
      else if (sim_exit != "0") functional_pass = 0
      else if (uvm_fatal == "-" || uvm_fatal == "?" || uvm_fatal == "") functional_pass = 0
      else if (uvm_error == "-" || uvm_error == "?" || uvm_error == "") functional_pass = 0
      else if (uvm_fatal + 0 != 0) functional_pass = 0
      else if (uvm_error + 0 != 0) functional_pass = 0

      if (!functional_pass) next

      bad = 0
      if (cov1 == "-" || cov1 == "?" || cov1 == "") bad = 1
      else if (cov2 == "-" || cov2 == "?" || cov2 == "") bad = 1
      else if ((cov1 + 0) < base) bad = 1
      else if ((cov2 + 0) < base) bad = 1

      if (bad) n += 1
    }
    END { print n + 0 }
  ' "$matrix" || true
)"

activity_fail_rows="$(
  awk -F'\t' -v min="$ACTIVITY_MIN_COVERAGE_PCT" '
    NR == 1 { next }
    {
      compile = $3
      sim_status = $5
      sim_exit = $6
      uvm_fatal = $9
      uvm_error = $10
      sim_log = $15

      functional_pass = 1
      if (compile != "OK") functional_pass = 0
      else if (sim_status != "OK") functional_pass = 0
      else if (sim_exit != "0") functional_pass = 0
      else if (uvm_fatal == "-" || uvm_fatal == "?" || uvm_fatal == "") functional_pass = 0
      else if (uvm_error == "-" || uvm_error == "?" || uvm_error == "") functional_pass = 0
      else if (uvm_fatal + 0 != 0) functional_pass = 0
      else if (uvm_error + 0 != 0) functional_pass = 0

      if (!functional_pass) next

      live = 0
      if (sim_log != "" && sim_log != "-") {
        while ((getline line < sim_log) > 0) {
          if (line ~ /[1-9][0-9]*[[:space:]]+hits([,[:space:]]|$)/) {
            live = 1
            break
          }

          if (match(line, /Overall coverage:[[:space:]]*[0-9]+(\.[0-9]+)?%/)) {
            cov_line = substr(line, RSTART, RLENGTH)
            gsub(/[^0-9.]/, "", cov_line)
            if ((cov_line + 0) > min) {
              live = 1
              break
            }
          }
        }
        close(sim_log)
      }

      if (!live) n += 1
    }
    END { print n + 0 }
  ' "$matrix" || true
)"

echo "[avip-circt-sim] gate-summary functional_fail_rows=$functional_fail_rows coverage_fail_rows=$coverage_fail_rows activity_fail_rows=$activity_fail_rows"
if [[ "$FAIL_ON_FUNCTIONAL_GATE" == "1" && "$functional_fail_rows" -gt 0 ]]; then
  echo "error: functional gate failed: $functional_fail_rows row(s)" >&2
  exit 1
fi
if [[ "$FAIL_ON_COVERAGE_BASELINE" == "1" && "$coverage_fail_rows" -gt 0 ]]; then
  echo "error: coverage baseline gate failed: $coverage_fail_rows row(s)" >&2
  exit 1
fi
if [[ "$FAIL_ON_ACTIVITY_LIVENESS" == "1" && "$activity_fail_rows" -gt 0 ]]; then
  echo "error: activity liveness gate failed: $activity_fail_rows row(s)" >&2
  exit 1
fi
