#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/thomas-ahle/circt/toy_models/open_source_formal_compare_20260301"
TOOLS="$ROOT/tools/install/bin"
LOG_DIR="$ROOT/logs/opentitan_realworld"
EQY_DIR="$ROOT/eqy_cases"
OUT_TSV="$ROOT/opentitan_realworld_oss.tsv"

mkdir -p "$LOG_DIR" "$EQY_DIR"
export PATH="$TOOLS:$PATH"

sanitize_pkg_files_for_yosys() {
  local work="$1"
  local case_id="$2"
  local out_list_file="$3"
  local sanitize_dir="$ROOT/.tmp_pkg_sanitized/${case_id}"

  rm -rf "$sanitize_dir"
  mkdir -p "$sanitize_dir"

  while IFS= read -r pkg; do
    local out
    out="$sanitize_dir/$(echo "$pkg" | sed 's#^/##; s#/#__#g')"
    # Yosys OSS parser currently rejects package-level import statements.
    # Strip them in a throwaway copy so we can observe deeper parser frontiers.
    sed -E \
      's/^([[:space:]]*)import[[:space:]]+[A-Za-z_][A-Za-z0-9_]*::\*;/\1\/\/ yosys-compat: stripped package import/' \
      "$pkg" > "$out"
    echo "$out"
  done < <(find "$work/fusesoc" -type f -name '*pkg.sv' | sort) > "$out_list_file"
}

run_case() {
  local task="$1"
  local tool="$2"
  local note="$3"
  local cmd="$4"
  local log="$LOG_DIR/${task}__${tool}.log"
  local start end ms rc result out

  start=$(date +%s%3N)
  set +e
  out=$(bash -lc "cd '$ROOT' && $cmd" 2>&1)
  rc=$?
  set -e
  end=$(date +%s%3N)
  ms=$((end - start))
  printf "%s\n" "$out" > "$log"

  result=$(printf "%s\n" "$out" | rg -o "DONE \((PASS|FAIL|ERROR), rc=[0-9]+\)|Successfully proved designs equivalent|Equivalence successfully proven!|ERROR: .*|syntax error.*|LEC_RESULT=[A-Z]+|UNKNOWN" | tail -n 1)
  if [[ -z "$result" ]]; then
    if [[ $rc -eq 0 ]]; then
      result="OK"
    else
      result="RC_${rc}"
    fi
  fi

  echo -e "${task}\t${tool}\t${rc}\t${ms}\t${result}\t${note}"
}

make_eqy_cfg() {
  local cfg="$1"
  local check="$2"
  local pkg_files="$3"
  local top="$4"
  local base="$5"
  cat > "$cfg" <<CFG
[options]

[gold]
read_verilog -sv $check $pkg_files $top
prep -top ${base}_ref
rename ${base}_ref top

[gate]
read_verilog -sv $check $pkg_files $top
prep -top ${base}_impl
rename ${base}_impl top

[strategy simple]
use sat
depth 1
CFG
}

emit_circt_row() {
  local task="$1"
  local results_tsv="$2"
  local rule="$3"
  local line
  line=$(grep -F "$rule" "$results_tsv" | head -n 1 || true)
  if [[ -n "$line" ]]; then
    local status result
    status=$(echo "$line" | cut -f1)
    result=$(echo "$line" | cut -f6)
    echo -e "${task}\tcirct\t0\t0\t${result}\t${status} (${rule}) [real-z3]"
  fi
}

{
  echo -e "task\ttool\trc\tms\tresult\tnote"

  declare -a CASES=(
    "aes|/home/thomas-ahle/circt/toy_models/opentitan_aes_lec_repro/work|__circt_conn_rule_0_CLKMGR_TRANS_AES|connectivity::clkmgr_trans.csv:CLKMGR_TRANS_AES"
    "flash_mode_o|/home/thomas-ahle/circt/toy_models/opentitan_flash_mode_o_repro/work|__circt_conn_rule_0_FLASH_TEST_MODE_O|connectivity::analog_sigs.csv:FLASH_TEST_MODE_O"
    "frontier_alert_handler_esc0|/home/thomas-ahle/circt/toy_models/opentitan_frontier_triplet_z3/work|__circt_conn_rule_0_ALERT_HANDLER_LC_CTRL_ESC0_RST|connectivity::alert_handler_esc.csv:ALERT_HANDLER_LC_CTRL_ESC0_RST"
    "frontier_ast_clk_es_in|/home/thomas-ahle/circt/toy_models/opentitan_frontier_triplet_z3/work|__circt_conn_rule_1_AST_CLK_ES_IN|connectivity::ast_clkmgr.csv:AST_CLK_ES_IN"
    "frontier_ast_hispeed_sel_in|/home/thomas-ahle/circt/toy_models/opentitan_frontier_triplet_z3/work|__circt_conn_rule_2_AST_HISPEED_SEL_IN|connectivity::ast_clkmgr.csv:AST_HISPEED_SEL_IN"
  )

  for entry in "${CASES[@]}"; do
    IFS='|' read -r case_id work base rule <<< "$entry"
    check="$work/checks/${base}.sv"
    pkg_list_file="$ROOT/.tmp_pkg_sanitized/${case_id}.list"
    sanitize_pkg_files_for_yosys "$work" "$case_id" "$pkg_list_file"
    pkg_files=$(tr '\n' ' ' < "$pkg_list_file")
    top=$(find "$work/fusesoc" -type f -name 'top_earlgrey.sv' | head -n 1)
    results_tsv="$(dirname "$work")/results.tsv"
    eqy_cfg="$EQY_DIR/${case_id}.eqy"

    run_case \
      "opentitan_${case_id}" \
      "yosys_oss" \
      "$rule" \
      "yosys -p \"read_verilog -sv $check $pkg_files $top; prep -top ${base}_impl\""

    make_eqy_cfg "$eqy_cfg" "$check" "$pkg_files" "$top" "$base"
    run_case \
      "opentitan_${case_id}" \
      "eqy_oss" \
      "$rule" \
      "rm -rf ${case_id}_eqy && eqy --yosys yosys -f -d ${case_id}_eqy '$eqy_cfg'"

    emit_circt_row "opentitan_${case_id}" "$results_tsv" "$rule"
  done

} > "$OUT_TSV"

echo "$OUT_TSV"
