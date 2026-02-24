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

CANONICAL_CIRCT_VERILOG="$CIRCT_ROOT/build-test/bin/circt-verilog"
CANONICAL_CIRCT_SIM="$CIRCT_ROOT/build-test/bin/circt-sim"

CIRCT_VERILOG="${CIRCT_VERILOG:-$CANONICAL_CIRCT_VERILOG}"
CIRCT_SIM="${CIRCT_SIM:-$CANONICAL_CIRCT_SIM}"
CIRCT_SIM_FALLBACK="${CIRCT_SIM_FALLBACK:-}"
RUN_AVIP="${RUN_AVIP:-$SCRIPT_DIR/run_avip_circt_verilog.sh}"
MBIT_DIR="${MBIT_DIR:-/home/thomas-ahle/mbit}"

AVIP_SET="${AVIP_SET:-core8}"
SEEDS_CSV="${SEEDS:-1}"
UVM_VERBOSITY="${UVM_VERBOSITY:-UVM_LOW}"
CIRCT_SIM_MODE="${CIRCT_SIM_MODE:-interpret}"
# AVIPs run UVM-heavy testbenches. Parallel simulation can introduce real host
# thread-level concurrency and break UVM's implicit single-thread assumptions,
# leading to spurious UVM_ERRORs (e.g. uvm_field_op state checks). Default to a
# single simulation thread; callers can override if desired.
CIRCT_SIM_EXTRA_ARGS="${CIRCT_SIM_EXTRA_ARGS:---parallel=1}"
CIRCT_SIM_WRITE_JIT_REPORT="${CIRCT_SIM_WRITE_JIT_REPORT:-0}"

MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-20}"
COMPILE_TIMEOUT="${COMPILE_TIMEOUT:-240}"
SIM_TIMEOUT="${SIM_TIMEOUT:-240}"
SIM_TIMEOUT_GRACE="${SIM_TIMEOUT_GRACE:-30}"
SIM_RETRIES="${SIM_RETRIES:-0}"
SIM_RETRY_ON_FCTTYP="${SIM_RETRY_ON_FCTTYP:-1}"
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
    echo "error: CIRCT_VERILOG must use canonical build-test path: $CANONICAL_CIRCT_VERILOG (got: $CIRCT_VERILOG)" >&2
    exit 1
  fi
  if [[ -n "${CIRCT_SIM:-}" && "$CIRCT_SIM" != "$CANONICAL_CIRCT_SIM" ]]; then
    echo "error: CIRCT_SIM must use canonical build-test path: $CANONICAL_CIRCT_SIM (got: $CIRCT_SIM)" >&2
    exit 1
  fi
fi

resolve_tool_path() {
  local requested="$1"
  local tool_name="$2"
  local candidates=("$requested")
  if [[ "$requested" == *"/build-test/"* ]]; then
    candidates+=("${requested/\/build-test\//\/build_test\/}")
  fi
  if [[ "$requested" == *"/build_test/"* ]]; then
    candidates+=("${requested/\/build_test\//\/build-test\/}")
  fi
  candidates+=(
    "$CIRCT_ROOT/build-test/bin/$tool_name"
    "$CIRCT_ROOT/build_test/bin/$tool_name"
    "$CIRCT_ROOT/build/bin/$tool_name"
  )

  local candidate=""
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  echo "$requested"
  return 0
}

CIRCT_VERILOG="$(resolve_tool_path "$CIRCT_VERILOG" circt-verilog)"
CIRCT_SIM="$(resolve_tool_path "$CIRCT_SIM" circt-sim)"
if [[ -z "$CIRCT_SIM_FALLBACK" ]]; then
  CIRCT_SIM_FALLBACK="$(resolve_tool_path "$CIRCT_ROOT/build-test/bin/circt-sim" circt-sim)"
fi

tool_help_healthy() {
  local tool="$1"
  [[ -x "$tool" ]] || return 1
  "$tool" --help >/dev/null 2>&1
}

if [[ "$CIRCT_ALLOW_NONCANONICAL_TOOLS" != "1" ]]; then
  if [[ -x "$CIRCT_SIM" ]] && ! tool_help_healthy "$CIRCT_SIM"; then
    if [[ -n "$CIRCT_SIM_FALLBACK" ]] && \
       [[ "$CIRCT_SIM_FALLBACK" != "$CIRCT_SIM" ]] && \
       tool_help_healthy "$CIRCT_SIM_FALLBACK"; then
      echo "warning: circt-sim probe failed for '$CIRCT_SIM'; falling back to '$CIRCT_SIM_FALLBACK'" >&2
      CIRCT_SIM="$CIRCT_SIM_FALLBACK"
    else
      echo "error: circt-sim probe failed for '$CIRCT_SIM' (rebuild that binary or set CIRCT_SIM_FALLBACK)" >&2
      exit 1
    fi
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

sim_extra_args=()
if [[ -n "$CIRCT_SIM_EXTRA_ARGS" ]]; then
  # Parsed as shell words; quote-preserving parsing is intentionally not used.
  read -r -a sim_extra_args <<< "$CIRCT_SIM_EXTRA_ARGS"
fi

run_limited() {
  local timeout_secs="$1"
  shift
  (
    ulimit -v "$MEMORY_LIMIT_KB" 2>/dev/null || true
    timeout --signal=KILL "$timeout_secs" "$@"
  )
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

  # Fallback: count individual occurrences. Return 0 if none; '?' is reserved
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
  if [[ "$SIM_RETRY_ON_FCTTYP" != "0" ]] && [[ "$code" -ne 0 ]] && \
     [[ -f "$log" ]] && grep -q "UVM_FATAL @ 0: FCTTYP" "$log"; then
    return 0
  fi
  return 1
}

matrix="$OUT_DIR/matrix.tsv"
cat > "$matrix" <<'HDR'
avip	seed	compile_status	compile_sec	sim_status	sim_exit	sim_sec	sim_time_fs	uvm_fatal	uvm_error	cov_1_pct	cov_2_pct	peak_rss_kb	compile_log	sim_log
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
  echo "fail_on_functional_gate=$FAIL_ON_FUNCTIONAL_GATE"
  echo "fail_on_coverage_baseline=$FAIL_ON_COVERAGE_BASELINE"
  echo "coverage_baseline_pct=$COVERAGE_BASELINE_PCT"
  echo "time_tool=${TIME_TOOL:-<none>}"
} > "$meta"

echo "[avip-circt-sim] out_dir=$OUT_DIR"
echo "[avip-circt-sim] selected_avips=${#selected_avips[@]} seeds=${#seeds[@]}"

for row in "${selected_avips[@]}"; do
  IFS='|' read -r name avip_dir filelist tops max_sim_fs test_name <<< "$row"
  avip_out="$OUT_DIR/$name"
  mkdir -p "$avip_out"

  compile_log="$avip_out/compile.log"
  mlir_file="$avip_out/${name}.mlir"

  compile_status="FAIL"
  compile_sec="-"

  if [[ ! -d "$avip_dir" ]]; then
    echo "warning: missing AVIP directory: $avip_dir" >&2
  else
    start_compile=$(date +%s)
    compile_cmd=("$RUN_AVIP" "$avip_dir")
    if [[ -n "$filelist" ]]; then
      compile_cmd+=("$filelist")
    fi
    if CIRCT_VERILOG="$CIRCT_VERILOG" \
       CIRCT_VERILOG_IR=llhd \
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

    sim_log="$avip_out/sim_seed_${seed}.log"
    rss_log="$avip_out/sim_seed_${seed}.rss_kb"
    jit_report="$avip_out/sim_seed_${seed}.jit-report.json"

    if [[ "$compile_status" == "OK" ]]; then
      top_flags=()
      IFS=',' read -ra top_modules <<< "$tops"
      for t in "${top_modules[@]}"; do
        t="${t//[[:space:]]/}"
        [[ -n "$t" ]] && top_flags+=(--top "$t")
      done

      uvm_args="+ntb_random_seed=$seed +UVM_VERBOSITY=$UVM_VERBOSITY"
      if [[ -n "$test_name" ]]; then
        uvm_args="+UVM_TESTNAME=$test_name $uvm_args"
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
            "${jit_report_flags[@]}" \
            --timeout="$SIM_TIMEOUT" \
            --max-time="$max_sim_fs" \
            --max-wall-ms="$MAX_WALL_MS" \
            > "$sim_log" 2>&1
        fi
        sim_exit=$?
        set -e

        if [[ "$sim_exit" -eq 0 ]]; then
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

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$name" "$seed" "$compile_status" "$compile_sec" "$sim_status" "$sim_exit" \
      "$sim_sec" "$sim_time_fs" "$uvm_fatal" "$uvm_error" "$cov_1" "$cov_2" \
      "$peak_rss_kb" "$compile_log" "$sim_log" >> "$matrix"

    echo "[avip-circt-sim] $name seed=$seed sim_status=$sim_status sim_exit=$sim_exit sim_sec=${sim_sec}s sim_time_fs=$sim_time_fs rss_kb=$peak_rss_kb"
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

echo "[avip-circt-sim] gate-summary functional_fail_rows=$functional_fail_rows coverage_fail_rows=$coverage_fail_rows"
if [[ "$FAIL_ON_FUNCTIONAL_GATE" == "1" && "$functional_fail_rows" -gt 0 ]]; then
  echo "error: functional gate failed: $functional_fail_rows row(s)" >&2
  exit 1
fi
if [[ "$FAIL_ON_COVERAGE_BASELINE" == "1" && "$coverage_fail_rows" -gt 0 ]]; then
  echo "error: coverage baseline gate failed: $coverage_fail_rows row(s)" >&2
  exit 1
fi
