#!/usr/bin/env bash
set -u

ROOT="/home/thomas-ahle/circt/toy_models/open_source_formal_compare_20260301"
TOOLS="$ROOT/tools/install/bin"
LOG_DIR="$ROOT/logs"
OUT_TSV="$ROOT/comparison_local_oss.tsv"

mkdir -p "$LOG_DIR"
export PATH="$TOOLS:$PATH"

{ 
  echo -e "task\ttool\trc\tms\tresult\tnote"

  run_case() {
    local task="$1"
    local tool="$2"
    local note="$3"
    local cmd="$4"
    local log="$LOG_DIR/${task}__${tool}.log"
    local start end ms rc result
    start=$(date +%s%3N)
    set +e
    local out
    out=$(bash -lc "cd '$ROOT' && $cmd" 2>&1)
    rc=$?
    set -e
    end=$(date +%s%3N)
    ms=$((end - start))
    printf "%s\n" "$out" > "$log"

    result=$(printf "%s\n" "$out" | rg -o "DONE \((PASS|FAIL|ERROR), rc=[0-9]+\)|Equivalence successfully proven!|BMC_RESULT=[A-Z]+|LEC_RESULT=[A-Z]+|ERROR: .*|syntax error.*|UNKNOWN" | tail -n 1)
    if [[ -z "$result" ]]; then
      if [[ $rc -eq 0 ]]; then
        result="OK"
      else
        result="RC_${rc}"
      fi
    fi

    echo -e "${task}\t${tool}\t${rc}\t${ms}\t${result}\t${note}"
  }

  run_case \
    "bmc_immediate" \
    "sby_oss" \
    "local sby+yosys+yosys-smtbmc" \
    "rm -rf bmc_immediate && sby -f bmc_immediate.sby"

  run_case \
    "bmc_concurrent_assert_property" \
    "sby_oss" \
    "assert property parser support" \
    "rm -rf bmc_basic_concurrent && sby -f bmc_basic_concurrent.sby"

  run_case \
    "bmc_yosys_basic00_pass" \
    "sby_oss" \
    "yosys/tests/sva/basic00 parity check (PASS profile)" \
    "rm -rf bmc_yosys_basic00_pass && sby -f bmc_yosys_basic00_pass.sby"

  run_case \
    "lec_simple" \
    "yosys_oss" \
    "equiv_make/equiv_simple/equiv_status" \
    "yosys -p \"read_verilog -sv lec_simple.sv; equiv_make modA modB equiv; prep -top equiv; equiv_simple; equiv_status -assert\""

  run_case \
    "lec_simple" \
    "eqy_oss" \
    "eqy simple strategy" \
    "rm -rf lec_simple && eqy -f lec_simple.eqy"

  run_case \
    "opentitan_aes_connectivity" \
    "yosys_oss" \
    "parse checker + top_earlgrey pkg/top files" \
    "yosys -p \"read_verilog -sv ../opentitan_aes_lec_repro/work/checks/__circt_conn_rule_0_CLKMGR_TRANS_AES.sv ../opentitan_aes_lec_repro/work/fusesoc/build/lowrisc_systems_chip_earlgrey_asic_0.1/src/lowrisc_systems_top_earlgrey_pkg_0.1/rtl/autogen/top_earlgrey_pkg.sv ../opentitan_aes_lec_repro/work/fusesoc/build/lowrisc_systems_chip_earlgrey_asic_0.1/src/lowrisc_systems_top_earlgrey_0.1/rtl/autogen/top_earlgrey.sv; prep -top __circt_conn_rule_0_CLKMGR_TRANS_AES_impl\""

  # Carry over known CIRCT rows for side-by-side context.
  if [[ -f "$ROOT/comparison.tsv" ]]; then
    awk -F '\t' 'NR > 1 && $2 == "circt" { print }' "$ROOT/comparison.tsv"
  fi

  # Carry over prior known-good CIRCT OpenTitan real-Z3 result for context.
  if [[ -s /home/thomas-ahle/circt/toy_models/opentitan_aes_lec_repro/results.tsv ]]; then
    local_line=$(head -n 1 /home/thomas-ahle/circt/toy_models/opentitan_aes_lec_repro/results.tsv)
    status=$(echo "$local_line" | cut -f1)
    case_name=$(echo "$local_line" | cut -f2)
    result=$(echo "$local_line" | cut -f6)
    echo -e "opentitan_aes_connectivity\tcirct\t0\t0\t${result}\t${status} (${case_name}) [real-z3]"
  fi
} > "$OUT_TSV"

echo "$OUT_TSV"
