#!/usr/bin/env bash
# Deterministic Xcelium AVIP matrix runner.
#
# Produces OUT_DIR/matrix.tsv with one row per (avip,seed), aligned with the
# circt-sim matrix schema for parity comparisons.
#
# Usage:
#   utils/run_avip_xcelium_reference.sh [out_dir]
#
# Key env vars:
#   AVIP_SET=core8|all9        (default: core8)
#   AVIPS=comma,list           (overrides AVIP_SET selection)
#   SEEDS=1,2,3                (default: 1)
#   MBIT_DIR=/home/.../mbit
#   COMPILE_TIMEOUT=600        seconds
#   SIM_TIMEOUT=300            seconds
#   MEMORY_LIMIT_GB=20

set -euo pipefail

OUT_DIR="${1:-/tmp/avip-xcelium-reference-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"

MBIT_DIR="${MBIT_DIR:-/home/thomas-ahle/mbit}"
AVIP_SET="${AVIP_SET:-core8}"
SEEDS_CSV="${SEEDS:-1}"
UVM_VERBOSITY="${UVM_VERBOSITY:-UVM_LOW}"

MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-20}"
COMPILE_TIMEOUT="${COMPILE_TIMEOUT:-600}"
SIM_TIMEOUT="${SIM_TIMEOUT:-300}"
MEMORY_LIMIT_KB=$((MEMORY_LIMIT_GB * 1024 * 1024))

TIME_TOOL=""
if [[ -x /usr/bin/time ]]; then
  TIME_TOOL="/usr/bin/time"
elif command -v gtime >/dev/null 2>&1; then
  TIME_TOOL="$(command -v gtime)"
fi

# name|cadence_dir|test_name
AVIPS_CORE8=(
  "apb|$MBIT_DIR/apb_avip/sim/cadence_sim|apb_8b_write_test"
  "ahb|$MBIT_DIR/ahb_avip/sim/cadenceSim|AhbWriteTest"
  "axi4|$MBIT_DIR/axi4_avip/sim/cadence_sim|axi4_write_read_test"
  "axi4Lite|$MBIT_DIR/axi4Lite_avip/sim/cadenceSim|MasterVIPSlaveIPWriteTest"
  "i2s|$MBIT_DIR/i2s_avip/sim/cadenceSim|I2sWriteOperationWith8bitdataTxMasterRxSlaveWith48khzTest"
  "i3c|$MBIT_DIR/i3c_avip/sim/cadence_sim|i3c_writeOperationWith8bitsData_test"
  "jtag|$MBIT_DIR/jtag_avip/sim/cadenceSim|JtagTdiWidth24Test"
  "spi|$MBIT_DIR/spi_avip/sim/cadenceSim|SpiSimpleFd8BitsTest"
)
AVIPS_ALL9=(
  "apb|$MBIT_DIR/apb_avip/sim/cadence_sim|apb_8b_write_test"
  "ahb|$MBIT_DIR/ahb_avip/sim/cadenceSim|AhbWriteTest"
  "axi4|$MBIT_DIR/axi4_avip/sim/cadence_sim|axi4_write_read_test"
  "axi4Lite|$MBIT_DIR/axi4Lite_avip/sim/cadenceSim|MasterVIPSlaveIPWriteTest"
  "i2s|$MBIT_DIR/i2s_avip/sim/cadenceSim|I2sWriteOperationWith8bitdataTxMasterRxSlaveWith48khzTest"
  "i3c|$MBIT_DIR/i3c_avip/sim/cadence_sim|i3c_writeOperationWith8bitsData_test"
  "jtag|$MBIT_DIR/jtag_avip/sim/cadenceSim|JtagTdiWidth24Test"
  "spi|$MBIT_DIR/spi_avip/sim/cadenceSim|SpiSimpleFd8BitsTest"
  "uart|$MBIT_DIR/uart_avip/sim/cadenceSim|UartBaudRate4800Test"
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
  val=$(grep -Eo "${key}[[:space:]]*:[[:space:]]*[0-9]+" "$log" | tail -1 | sed -E 's/.*:[[:space:]]*([0-9]+)/\1/' || true)
  [[ -n "$val" ]] && echo "$val" || echo "?"
}

extract_coverage_pair() {
  local log="$1"
  local values
  values=$(grep -Eo '(Total[[:space:]]+)?Coverage[[:space:]]*=[[:space:]]*[0-9]+(\.[0-9]+)?[[:space:]]*%?' "$log" \
      | sed -E 's/.*=[[:space:]]*([0-9]+(\.[0-9]+)?)[[:space:]]*%?/\1/' \
      | tr '\n' ' ' || true)
  local cov1="-"
  local cov2="-"
  if [[ -n "$values" ]]; then
    read -r cov1 cov2 _ <<< "$values"
    [[ -z "$cov1" ]] && cov1="-"
    [[ -z "$cov2" ]] && cov2="-"
  fi
  echo "$cov1|$cov2"
}

extract_sim_time_raw() {
  local log="$1"
  local raw
  raw=$(grep -Eoi 'at time[[:space:]]+[0-9]+(\.[0-9]+)?[[:space:]]*(fs|ps|ns|us|ms|s)' "$log" \
      | tail -1 \
      | tr '[:upper:]' '[:lower:]' \
      | sed -E 's/.*time[[:space:]]+([0-9]+(\.[0-9]+)?)[[:space:]]*(fs|ps|ns|us|ms|s)/\1 \3/' || true)
  if [[ -z "$raw" ]]; then
    raw=$(grep -Eoi '\$finish[^\n]*[0-9]+(\.[0-9]+)?[[:space:]]*(fs|ps|ns|us|ms|s)' "$log" \
        | tail -1 \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/.*([0-9]+(\.[0-9]+)?)[[:space:]]*(fs|ps|ns|us|ms|s).*/\1 \3/' || true)
  fi
  [[ -n "$raw" ]] && echo "$raw" || echo "-"
}

to_fs() {
  local value="$1"
  local unit="$2"
  awk -v v="$value" -v u="$unit" 'BEGIN {
    m = 1;
    if (u == "fs") m = 1;
    else if (u == "ps") m = 1e3;
    else if (u == "ns") m = 1e6;
    else if (u == "us") m = 1e9;
    else if (u == "ms") m = 1e12;
    else if (u == "s") m = 1e15;
    else { print "-"; exit 0; }
    printf "%.0f", v * m;
  }'
}

extract_sim_time_fs() {
  local log="$1"
  local raw
  raw=$(extract_sim_time_raw "$log")
  if [[ "$raw" == "-" ]]; then
    echo "-"
    return
  fi
  local value unit
  read -r value unit <<< "$raw"
  unit="${unit,,}"
  if [[ -z "$value" || -z "$unit" ]]; then
    echo "-"
    return
  fi
  to_fs "$value" "$unit"
}

sim_status_from_exit() {
  local code="$1"
  case "$code" in
    0) echo "OK" ;;
    124|137) echo "TIMEOUT" ;;
    *) echo "FAIL" ;;
  esac
}

matrix="$OUT_DIR/matrix.tsv"
cat > "$matrix" <<'HDR'
avip	seed	compile_status	compile_sec	sim_status	sim_exit	sim_sec	sim_time_raw	sim_time_fs	uvm_fatal	uvm_error	cov_1_pct	cov_2_pct	peak_rss_kb	compile_log	sim_log
HDR

meta="$OUT_DIR/meta.txt"
{
  echo "generated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "mbit_dir=$MBIT_DIR"
  echo "avip_set=$AVIP_SET"
  echo "avips=${AVIPS:-<from-set>}"
  echo "seeds=$SEEDS_CSV"
  echo "memory_limit_gb=$MEMORY_LIMIT_GB"
  echo "compile_timeout=$COMPILE_TIMEOUT"
  echo "sim_timeout=$SIM_TIMEOUT"
  echo "time_tool=${TIME_TOOL:-<none>}"
} > "$meta"

echo "[avip-xcelium] out_dir=$OUT_DIR"
echo "[avip-xcelium] selected_avips=${#selected_avips[@]} seeds=${#seeds[@]}"

for row in "${selected_avips[@]}"; do
  IFS='|' read -r name caddir test_name <<< "$row"
  avip_out="$OUT_DIR/$name"
  mkdir -p "$avip_out"

  compile_log="$avip_out/compile.log"
  compile_status="FAIL"
  compile_sec="-"

  if [[ ! -d "$caddir" ]]; then
    echo "warning: missing cadence directory: $caddir" >&2
  else
    start_compile=$(date +%s)
    set +e
    (
      cd "$caddir"
      run_limited "$COMPILE_TIMEOUT" make compile
    ) > "$compile_log" 2>&1
    c_exit=$?
    set -e
    end_compile=$(date +%s)
    compile_sec=$((end_compile - start_compile))
    if [[ $c_exit -eq 0 ]]; then
      compile_status="OK"
    fi
  fi

  echo "[avip-xcelium] $name compile_status=$compile_status compile_sec=${compile_sec}s"

  for seed in "${seeds[@]}"; do
    seed="${seed//[[:space:]]/}"
    [[ -z "$seed" ]] && continue

    sim_status="SKIP"
    sim_exit="-"
    sim_sec="-"
    sim_time_raw="-"
    sim_time_fs="-"
    uvm_fatal="-"
    uvm_error="-"
    cov_1="-"
    cov_2="-"
    peak_rss_kb="-"

    sim_wrapper_log="$avip_out/sim_seed_${seed}.wrapper.log"
    sim_log="$avip_out/sim_seed_${seed}.log"
    rss_log="$avip_out/sim_seed_${seed}.rss_kb"

    if [[ "$compile_status" == "OK" ]]; then
      start_sim=$(date +%s)
      set +e
      if [[ -n "$TIME_TOOL" ]]; then
        (
          cd "$caddir"
          rm -rf "$test_name" >/dev/null 2>&1 || true
          run_limited "$SIM_TIMEOUT" \
            "$TIME_TOOL" -f "%M" -o "$rss_log" \
            make simulate test="$test_name" seed="$seed" uvm_verbosity="$UVM_VERBOSITY"
        ) > "$sim_wrapper_log" 2>&1
      else
        (
          cd "$caddir"
          rm -rf "$test_name" >/dev/null 2>&1 || true
          run_limited "$SIM_TIMEOUT" \
            make simulate test="$test_name" seed="$seed" uvm_verbosity="$UVM_VERBOSITY"
        ) > "$sim_wrapper_log" 2>&1
      fi
      sim_exit=$?
      set -e
      end_sim=$(date +%s)
      sim_sec=$((end_sim - start_sim))

      # Copy canonical test log if present, otherwise use wrapper log.
      if [[ -f "$caddir/$test_name/$test_name.log" ]]; then
        cp "$caddir/$test_name/$test_name.log" "$sim_log"
      else
        cp "$sim_wrapper_log" "$sim_log"
      fi

      sim_status=$(sim_status_from_exit "$sim_exit")

      if [[ -f "$rss_log" ]]; then
        peak_rss_kb=$(tail -n 1 "$rss_log" | tr -d '[:space:]')
        [[ -z "$peak_rss_kb" ]] && peak_rss_kb="-"
      fi

      sim_time_raw=$(extract_sim_time_raw "$sim_log")
      sim_time_fs=$(extract_sim_time_fs "$sim_log")
      uvm_fatal=$(extract_uvm_count "UVM_FATAL" "$sim_log")
      uvm_error=$(extract_uvm_count "UVM_ERROR" "$sim_log")
      cov_pair=$(extract_coverage_pair "$sim_log")
      IFS='|' read -r cov_1 cov_2 <<< "$cov_pair"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$name" "$seed" "$compile_status" "$compile_sec" "$sim_status" "$sim_exit" \
      "$sim_sec" "$sim_time_raw" "$sim_time_fs" "$uvm_fatal" "$uvm_error" \
      "$cov_1" "$cov_2" "$peak_rss_kb" "$compile_log" "$sim_log" >> "$matrix"

    echo "[avip-xcelium] $name seed=$seed sim_status=$sim_status sim_exit=$sim_exit sim_time_raw='$sim_time_raw' sim_time_fs=$sim_time_fs rss_kb=$peak_rss_kb"
  done
done

echo "[avip-xcelium] matrix=$matrix"
echo "[avip-xcelium] meta=$meta"
