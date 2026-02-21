#!/usr/bin/env bash
# Deterministic AVIP matrix runner using arcilator behavioral mode.
#
# Produces a per-(avip,seed) matrix at OUT_DIR/matrix.tsv.
#
# Usage:
#   utils/run_avip_arcilator_sim.sh [out_dir]
#
# Key env vars:
#   AVIP_SET=core8|all9        (default: core8)
#   AVIPS=comma,list           (overrides AVIP_SET selection)
#   SEEDS=1,2,3                (default: 1)
#   MBIT_DIR=/home/.../mbit
#   COMPILE_TIMEOUT=300        seconds
#   SIM_TIMEOUT=180            seconds
#   SIM_TIMEOUT_GRACE=30       extra seconds before external hard-kill
#   MEMORY_LIMIT_GB=20
#   MAX_WALL_MS=180000
#   UVM_VERBOSITY=UVM_LOW|UVM_HIGH
#   CIRCT_VERILOG_ARGS="..."   extra args for circt-verilog
#   ARCILATOR_ARGS="..."       extra args for arcilator

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIRCT_ROOT="${CIRCT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUT_DIR="${1:-/tmp/avip-arcilator-sim-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"

CIRCT_VERILOG="${CIRCT_VERILOG:-$CIRCT_ROOT/build-test/bin/circt-verilog}"
ARCILATOR="${ARCILATOR:-$CIRCT_ROOT/build-test/bin/arcilator}"
RUN_AVIP="${RUN_AVIP:-$SCRIPT_DIR/run_avip_circt_verilog.sh}"
MBIT_DIR="${MBIT_DIR:-/home/thomas-ahle/mbit}"

AVIP_SET="${AVIP_SET:-core8}"
SEEDS_CSV="${SEEDS:-1}"
UVM_VERBOSITY="${UVM_VERBOSITY:-UVM_LOW}"
ARCILATOR_FAST_MODE="${ARCILATOR_FAST_MODE:-1}"

MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-20}"
COMPILE_TIMEOUT="${COMPILE_TIMEOUT:-300}"
SIM_TIMEOUT="${SIM_TIMEOUT:-180}"
SIM_TIMEOUT_GRACE="${SIM_TIMEOUT_GRACE:-30}"
SIM_TIMEOUT_HARD=$((SIM_TIMEOUT + SIM_TIMEOUT_GRACE))
if [[ -z "${MAX_WALL_MS+x}" ]]; then
  MAX_WALL_MS="$((SIM_TIMEOUT_HARD * 1000))"
fi
MEMORY_LIMIT_KB=$((MEMORY_LIMIT_GB * 1024 * 1024))

CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"
ARCILATOR_ARGS="${ARCILATOR_ARGS:-}"

# name|avip_dir|filelist|tops|test_name
AVIPS_CORE8=(
  "apb|$MBIT_DIR/apb_avip|$MBIT_DIR/apb_avip/sim/apb_compile.f|hdl_top,hvl_top|apb_8b_write_test"
  "ahb|$MBIT_DIR/ahb_avip|$MBIT_DIR/ahb_avip/sim/ahb_compile.f|HdlTop,HvlTop|AhbWriteTest"
  "axi4|$MBIT_DIR/axi4_avip|$MBIT_DIR/axi4_avip/sim/axi4_compile.f|hdl_top,hvl_top|axi4_write_read_test"
  "axi4Lite|$MBIT_DIR/axi4Lite_avip|$MBIT_DIR/axi4Lite_avip/sim/Axi4LiteProject.f|Axi4LiteHdlTop,Axi4LiteHvlTop|Axi4LiteWriteTest"
  "i2s|$MBIT_DIR/i2s_avip|$MBIT_DIR/i2s_avip/sim/I2sCompile.f|hdlTop,hvlTop|I2sWriteOperationWith8bitdataTxMasterRxSlaveWith48khzTest"
  "i3c|$MBIT_DIR/i3c_avip|$MBIT_DIR/i3c_avip/sim/i3c_compile.f|hdl_top,hvl_top|i3c_writeOperationWith8bitsData_test"
  "jtag|$MBIT_DIR/jtag_avip|$MBIT_DIR/jtag_avip/sim/JtagCompile.f|HdlTop,HvlTop|JtagTdiWidth24Test"
  "spi|$MBIT_DIR/spi_avip|$MBIT_DIR/spi_avip/sim/SpiCompile.f|SpiHdlTop,SpiHvlTop|SpiSimpleFd8BitsTest"
)
AVIPS_ALL9=(
  "apb|$MBIT_DIR/apb_avip|$MBIT_DIR/apb_avip/sim/apb_compile.f|hdl_top,hvl_top|apb_8b_write_test"
  "ahb|$MBIT_DIR/ahb_avip|$MBIT_DIR/ahb_avip/sim/ahb_compile.f|HdlTop,HvlTop|AhbWriteTest"
  "axi4|$MBIT_DIR/axi4_avip|$MBIT_DIR/axi4_avip/sim/axi4_compile.f|hdl_top,hvl_top|axi4_write_read_test"
  "axi4Lite|$MBIT_DIR/axi4Lite_avip|$MBIT_DIR/axi4Lite_avip/sim/Axi4LiteProject.f|Axi4LiteHdlTop,Axi4LiteHvlTop|Axi4LiteWriteTest"
  "i2s|$MBIT_DIR/i2s_avip|$MBIT_DIR/i2s_avip/sim/I2sCompile.f|hdlTop,hvlTop|I2sWriteOperationWith8bitdataTxMasterRxSlaveWith48khzTest"
  "i3c|$MBIT_DIR/i3c_avip|$MBIT_DIR/i3c_avip/sim/i3c_compile.f|hdl_top,hvl_top|i3c_writeOperationWith8bitsData_test"
  "jtag|$MBIT_DIR/jtag_avip|$MBIT_DIR/jtag_avip/sim/JtagCompile.f|HdlTop,HvlTop|JtagTdiWidth24Test"
  "spi|$MBIT_DIR/spi_avip|$MBIT_DIR/spi_avip/sim/SpiCompile.f|SpiHdlTop,SpiHvlTop|SpiSimpleFd8BitsTest"
  "uart|$MBIT_DIR/uart_avip|$MBIT_DIR/uart_avip/sim/UartCompile.f|HdlTop,HvlTop|UartBaudRate4800Test"
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

if [[ ! -x "$CIRCT_VERILOG" ]]; then
  echo "error: circt-verilog not found or not executable: $CIRCT_VERILOG" >&2
  exit 1
fi
if [[ ! -x "$ARCILATOR" ]]; then
  echo "error: arcilator not found or not executable: $ARCILATOR" >&2
  exit 1
fi
if [[ ! -x "$RUN_AVIP" ]]; then
  echo "error: helper runner not found or not executable: $RUN_AVIP" >&2
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
  val=$(grep -E "^[[:space:]]*${key}[[:space:]]*:[[:space:]]*[0-9]+([[:space:]]|$)" "$log" \
      | tail -1 | sed -E 's/.*:[[:space:]]*([0-9]+).*/\1/' || true)
  if [[ -n "$val" ]]; then
    echo "$val"
    return
  fi
  local count
  count=$( (grep -Eo "${key}" "$log" || true) | wc -l | tr -d '[:space:]' )
  echo "$count"
}

extract_sim_time_fs() {
  local log="$1"
  local val
  val=$(grep -Eo 'Simulation terminated at time[[:space:]]+[0-9]+[[:space:]]+fs' "$log" \
      | tail -1 | sed -E 's/.*time[[:space:]]+([0-9]+)[[:space:]]+fs/\1/' || true)
  if [[ -z "$val" ]]; then
    val=$(grep -Eo 'Simulation completed at time[[:space:]]+[0-9]+[[:space:]]+fs' "$log" \
        | tail -1 | sed -E 's/.*time[[:space:]]+([0-9]+)[[:space:]]+fs/\1/' || true)
  fi
  [[ -n "$val" ]] && echo "$val" || echo "-"
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
avip	seed	compile_status	compile_sec	sim_status	sim_exit	sim_sec	sim_time_fs	uvm_fatal	uvm_error	compile_log	sim_log
HDR

meta="$OUT_DIR/meta.txt"
git_sha="$(git -C "$CIRCT_ROOT" rev-parse HEAD 2>/dev/null || echo "<unknown>")"
{
  echo "generated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "circt_root=$CIRCT_ROOT"
  echo "git_sha=$git_sha"
  echo "script_path=$SCRIPT_DIR/run_avip_arcilator_sim.sh"
  echo "circt_verilog=$CIRCT_VERILOG"
  echo "arcilator=$ARCILATOR"
  echo "avip_set=$AVIP_SET"
  echo "avips=${AVIPS:-<from-set>}"
  echo "seeds=$SEEDS_CSV"
  echo "memory_limit_gb=$MEMORY_LIMIT_GB"
  echo "compile_timeout=$COMPILE_TIMEOUT"
  echo "sim_timeout=$SIM_TIMEOUT"
  echo "sim_timeout_grace=$SIM_TIMEOUT_GRACE"
  echo "sim_timeout_hard=$SIM_TIMEOUT_HARD"
  echo "max_wall_ms=$MAX_WALL_MS"
  echo "uvm_verbosity=$UVM_VERBOSITY"
  echo "arcilator_fast_mode=$ARCILATOR_FAST_MODE"
  echo "arcilator_strip_global_ctors=${ARCILATOR_STRIP_GLOBAL_CTORS:-<auto>}"
  echo "circt_verilog_args=${CIRCT_VERILOG_ARGS:-<none>}"
  echo "arcilator_args=${ARCILATOR_ARGS:-<none>}"
} > "$meta"

echo "[avip-arcilator-sim] out_dir=$OUT_DIR"
echo "[avip-arcilator-sim] selected_avips=${#selected_avips[@]} seeds=${#seeds[@]}"

arcilator_extra_args=()
if [[ -n "$ARCILATOR_ARGS" ]]; then
  read -r -a arcilator_extra_args <<< "$ARCILATOR_ARGS"
fi

for row in "${selected_avips[@]}"; do
  IFS='|' read -r name avip_dir filelist tops test_name <<< "$row"
  IFS=',' read -ra top_modules <<< "$tops"
  lane_jit_entry=""
  if [[ "$ARCILATOR_FAST_MODE" == "1" ]]; then
    for t in "${top_modules[@]}"; do
      t="${t//[[:space:]]/}"
      [[ -z "$t" ]] && continue
      lower_t="${t,,}"
      if [[ "$lower_t" == *hvl*top* ]]; then
        lane_jit_entry="$t"
        break
      fi
    done
  fi
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
    lane_verilog_args="$CIRCT_VERILOG_ARGS"
    if [[ "$name" == "jtag" ]]; then
      jtag_compat_args="--allow-virtual-iface-with-override --relax-enum-conversions --compat=all"
      if [[ -n "$lane_verilog_args" ]]; then
        lane_verilog_args="$lane_verilog_args $jtag_compat_args"
      else
        lane_verilog_args="$jtag_compat_args"
      fi
    fi
    for t in "${top_modules[@]}"; do
      t="${t//[[:space:]]/}"
      [[ -z "$t" ]] && continue
      lane_verilog_args="${lane_verilog_args:+$lane_verilog_args }--top $t"
    done

    compile_cmd=("$RUN_AVIP" "$avip_dir")
    if [[ -n "$filelist" ]]; then
      compile_cmd+=("$filelist")
    fi
    if CIRCT_VERILOG="$CIRCT_VERILOG" \
       CIRCT_VERILOG_IR=llhd \
       CIRCT_VERILOG_ARGS="$lane_verilog_args" \
       OUT="$mlir_file" \
       run_limited "$COMPILE_TIMEOUT" "${compile_cmd[@]}" > "$compile_log" 2>&1; then
      compile_status="OK"
    fi
    end_compile=$(date +%s)
    compile_sec=$((end_compile - start_compile))
  fi

  echo "[avip-arcilator-sim] $name compile_status=$compile_status compile_sec=${compile_sec}s"

  for seed in "${seeds[@]}"; do
    seed="${seed//[[:space:]]/}"
    [[ -z "$seed" ]] && continue

    sim_status="SKIP"
    sim_exit="-"
    sim_sec="-"
    sim_time_fs="-"
    uvm_fatal="-"
    uvm_error="-"
    sim_log="$avip_out/sim_seed_${seed}.log"

    if [[ "$compile_status" == "OK" ]]; then
      uvm_args="+ntb_random_seed=$seed +UVM_VERBOSITY=$UVM_VERBOSITY"
      if [[ -n "$test_name" ]]; then
        uvm_args="+UVM_TESTNAME=$test_name $uvm_args"
      fi

      start_sim=$(date +%s)
      lane_arcilator_args=("${arcilator_extra_args[@]}")
      has_jit_entry_arg=0
      for arg in "${lane_arcilator_args[@]}"; do
        if [[ "$arg" == "--jit-entry" || "$arg" == --jit-entry=* ]]; then
          has_jit_entry_arg=1
          break
        fi
      done
      if [[ "$ARCILATOR_FAST_MODE" == "1" && $has_jit_entry_arg -eq 0 && -n "$lane_jit_entry" ]]; then
        lane_arcilator_args+=("--jit-entry=$lane_jit_entry")
      fi

      strip_global_ctors="${ARCILATOR_STRIP_GLOBAL_CTORS:-$ARCILATOR_FAST_MODE}"
      set +e
      run_limited "$SIM_TIMEOUT_HARD" \
        env CIRCT_UVM_ARGS="$uvm_args" ARCILATOR_STRIP_GLOBAL_CTORS="$strip_global_ctors" \
        "$ARCILATOR" \
        --behavioral \
        --run \
        "${lane_arcilator_args[@]}" \
        --max-wall-ms="$MAX_WALL_MS" \
        "$mlir_file" > "$sim_log" 2>&1
      sim_exit=$?
      set -e
      end_sim=$(date +%s)
      sim_sec=$((end_sim - start_sim))

      sim_status=$(sim_status_from_exit "$sim_exit")
      sim_time_fs=$(extract_sim_time_fs "$sim_log")
      uvm_fatal=$(extract_uvm_count "UVM_FATAL" "$sim_log")
      uvm_error=$(extract_uvm_count "UVM_ERROR" "$sim_log")
      if [[ "$sim_status" == "OK" ]]; then
        if [[ "$uvm_fatal" =~ ^[0-9]+$ ]] && (( uvm_fatal > 0 )); then
          sim_status="FAIL"
        fi
        if [[ "$uvm_error" =~ ^[0-9]+$ ]] && (( uvm_error > 0 )); then
          sim_status="FAIL"
        fi
      fi
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$name" "$seed" "$compile_status" "$compile_sec" "$sim_status" "$sim_exit" \
      "$sim_sec" "$sim_time_fs" "$uvm_fatal" "$uvm_error" "$compile_log" \
      "$sim_log" >> "$matrix"

    echo "[avip-arcilator-sim] $name seed=$seed sim_status=$sim_status sim_exit=$sim_exit sim_sec=${sim_sec}s sim_time_fs=$sim_time_fs"
  done
done

echo "[avip-arcilator-sim] matrix=$matrix"
echo "[avip-arcilator-sim] meta=$meta"
