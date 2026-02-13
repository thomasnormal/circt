#!/usr/bin/env bash
set -euo pipefail

VERIF_DIR="${1:-/home/thomas-ahle/verilator-verification}"
shift || true
BOUND="${BOUND:-10}"

resolve_default_circt_tool() {
  local tool_name="$1"
  local preferred_dir="${2:-}"
  local candidate

  if [[ -n "$preferred_dir" ]]; then
    candidate="${preferred_dir}/${tool_name}"
    if [[ -x "$candidate" ]]; then
      printf "%s\n" "$candidate"
      return 0
    fi
  fi

  for candidate in "build/bin/${tool_name}" "build-test/bin/${tool_name}"; do
    if [[ -x "$candidate" ]]; then
      printf "%s\n" "$candidate"
      return 0
    fi
  done

  if command -v "$tool_name" >/dev/null 2>&1; then
    command -v "$tool_name"
    return 0
  fi

  if [[ -n "$preferred_dir" ]]; then
    printf "%s\n" "${preferred_dir}/${tool_name}"
  else
    printf "build/bin/%s\n" "$tool_name"
  fi
}

derive_tool_dir_from_verilog() {
  local verilog_tool="$1"
  local resolved=""

  if [[ "$verilog_tool" == */* ]]; then
    printf "%s\n" "$(dirname "$verilog_tool")"
    return 0
  fi

  resolved="$(command -v "$verilog_tool" 2>/dev/null || true)"
  if [[ -n "$resolved" ]]; then
    printf "%s\n" "$(dirname "$resolved")"
    return 0
  fi

  printf "%s\n" "build/bin"
}

# Memory limit settings to prevent system hangs
CIRCT_MEMORY_LIMIT_GB="${CIRCT_MEMORY_LIMIT_GB:-20}"
CIRCT_TIMEOUT_SECS="${CIRCT_TIMEOUT_SECS:-300}"
CIRCT_MEMORY_LIMIT_KB=$((CIRCT_MEMORY_LIMIT_GB * 1024 * 1024))

# Run a command with memory limit
run_limited() {
  (
    ulimit -v $CIRCT_MEMORY_LIMIT_KB 2>/dev/null || true
    timeout --signal=KILL $CIRCT_TIMEOUT_SECS "$@"
  )
}
IGNORE_ASSERTS_UNTIL="${IGNORE_ASSERTS_UNTIL:-1}"
RISING_CLOCKS_ONLY="${RISING_CLOCKS_ONLY:-0}"
ALLOW_MULTI_CLOCK="${ALLOW_MULTI_CLOCK:-0}"
Z3_LIB="${Z3_LIB:-/home/thomas-ahle/z3-install/lib64/libz3.so}"
CIRCT_VERILOG="${CIRCT_VERILOG:-$(resolve_default_circt_tool "circt-verilog")}"
CIRCT_TOOL_DIR_DEFAULT="$(derive_tool_dir_from_verilog "$CIRCT_VERILOG")"
CIRCT_BMC="${CIRCT_BMC:-$(resolve_default_circt_tool "circt-bmc" "$CIRCT_TOOL_DIR_DEFAULT")}"
CIRCT_BMC_ARGS="${CIRCT_BMC_ARGS:-}"
BMC_SMOKE_ONLY="${BMC_SMOKE_ONLY:-0}"
BMC_FAIL_ON_VIOLATION="${BMC_FAIL_ON_VIOLATION:-1}"
BMC_RUN_SMTLIB="${BMC_RUN_SMTLIB:-0}"
KEEP_LOGS_DIR="${KEEP_LOGS_DIR:-}"
BMC_ABSTRACTION_PROVENANCE_OUT="${BMC_ABSTRACTION_PROVENANCE_OUT:-}"
BMC_CHECK_ATTRIBUTION_OUT="${BMC_CHECK_ATTRIBUTION_OUT:-}"
BMC_DROP_REMARK_CASES_OUT="${BMC_DROP_REMARK_CASES_OUT:-}"
BMC_DROP_REMARK_REASONS_OUT="${BMC_DROP_REMARK_REASONS_OUT:-}"
BMC_SEMANTIC_TAG_MAP_FILE="${BMC_SEMANTIC_TAG_MAP_FILE:-}"
# NOTE: NO_PROPERTY_AS_SKIP defaults to 0 because the "no property provided to check"
# warning is SPURIOUS for clocked assertions that are lowered later in the pipeline.
# Setting this to 1 would cause false SKIP results for otherwise valid tests.
NO_PROPERTY_AS_SKIP="${NO_PROPERTY_AS_SKIP:-0}"
TOP="${TOP:-top}"
TEST_FILTER="${TEST_FILTER:-}"
OUT="${OUT:-$PWD/verilator-verification-bmc-results.txt}"
XFAILS="${XFAILS:-}"
DISABLE_UVM_AUTO_INCLUDE="${DISABLE_UVM_AUTO_INCLUDE:-1}"
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"
Z3_BIN="${Z3_BIN:-}"
BMC_ASSUME_KNOWN_INPUTS="${BMC_ASSUME_KNOWN_INPUTS:-0}"
FAIL_ON_DROP_REMARKS="${FAIL_ON_DROP_REMARKS:-0}"
DROP_REMARK_PATTERN="${DROP_REMARK_PATTERN:-will be dropped during lowering}"

if [[ ! -d "$VERIF_DIR/tests" ]]; then
  echo "verilator-verification directory not found: $VERIF_DIR" >&2
  exit 1
fi

if [[ -z "$TEST_FILTER" ]]; then
  echo "must set TEST_FILTER explicitly (no default filter)" >&2
  exit 1
fi

if [[ "$BMC_RUN_SMTLIB" == "1" && "$BMC_SMOKE_ONLY" != "1" ]]; then
  if [[ -z "$Z3_BIN" ]]; then
    if command -v z3 >/dev/null 2>&1; then
      Z3_BIN="z3"
    elif [[ -x /home/thomas-ahle/z3-install/bin/z3 ]]; then
      Z3_BIN="/home/thomas-ahle/z3-install/bin/z3"
    elif [[ -x /home/thomas-ahle/z3/build/z3 ]]; then
      Z3_BIN="/home/thomas-ahle/z3/build/z3"
    fi
  fi
  if [[ -z "$Z3_BIN" ]]; then
    echo "z3 not found; set Z3_BIN or disable BMC_RUN_SMTLIB" >&2
    exit 1
  fi
fi

suites=("$@")
if [[ ${#suites[@]} -eq 0 ]]; then
  suites=("$VERIF_DIR/tests/asserts" "$VERIF_DIR/tests/sequences" \
    "$VERIF_DIR/tests/event-control-expression")
else
  for i in "${!suites[@]}"; do
    if [[ "${suites[$i]}" != /* ]]; then
      suites[$i]="$VERIF_DIR/${suites[$i]}"
    fi
  done
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
unknown=0
timeout=0
skip=0
total=0
drop_remark_cases=0
declare -A semantic_tags_by_case
declare -A drop_remark_seen_cases
declare -A drop_remark_seen_case_reasons

is_xfail() {
  local name="$1"
  if [[ -z "$XFAILS" ]]; then
    return 1
  fi
  if [[ ! -f "$XFAILS" ]]; then
    return 1
  fi
  grep -Fxq "$name" "$XFAILS"
}

load_semantic_tag_map() {
  if [[ -z "$BMC_SEMANTIC_TAG_MAP_FILE" || ! -f "$BMC_SEMANTIC_TAG_MAP_FILE" ]]; then
    return
  fi
  while IFS=$'\t' read -r case_name tags extra; do
    if [[ -z "$case_name" || "$case_name" =~ ^# ]]; then
      continue
    fi
    if [[ -n "${extra:-}" ]]; then
      continue
    fi
    tags="${tags// /}"
    if [[ -z "$tags" ]]; then
      continue
    fi
    semantic_tags_by_case["$case_name"]="$tags"
  done < "$BMC_SEMANTIC_TAG_MAP_FILE"
}

emit_result_row() {
  local status="$1"
  local base="$2"
  local sv="$3"
  local tags="${semantic_tags_by_case[$base]-}"
  if [[ -n "$tags" ]]; then
    printf "%s\t%s\t%s\tverilator-verification\tBMC\tsemantic_buckets=%s\n" \
      "$status" "$base" "$sv" "$tags" >> "$results_tmp"
  else
    printf "%s\t%s\t%s\n" "$status" "$base" "$sv" >> "$results_tmp"
  fi
}

record_drop_remark_case() {
  local case_id="$1"
  local case_path="$2"
  local verilog_log="$3"
  if [[ -z "$case_id" || ! -s "$verilog_log" ]]; then
    return
  fi
  if ! grep -Fq "$DROP_REMARK_PATTERN" "$verilog_log"; then
    return
  fi
  while IFS= read -r reason; do
    if [[ -z "$reason" ]]; then
      continue
    fi
    local reason_key="${case_id}|${reason}"
    if [[ -n "${drop_remark_seen_case_reasons["$reason_key"]+x}" ]]; then
      continue
    fi
    drop_remark_seen_case_reasons["$reason_key"]=1
    if [[ -n "$BMC_DROP_REMARK_REASONS_OUT" ]]; then
      printf "%s\t%s\t%s\n" "$case_id" "$case_path" "$reason" >> "$BMC_DROP_REMARK_REASONS_OUT"
    fi
  done < <(
    awk -v pattern="$DROP_REMARK_PATTERN" '
      index($0, pattern) {
        line = $0
        gsub(/\t/, " ", line)
        sub(/^[[:space:]]+/, "", line)
        if (match(line, /^[^:]+:[0-9]+(:[0-9]+)?:[[:space:]]*/))
          line = substr(line, RLENGTH + 1)
        sub(/^[Ww]arning:[[:space:]]*/, "", line)
        gsub(/[[:space:]]+/, " ", line)
        gsub(/[0-9]+/, "<n>", line)
        gsub(/;/, ",", line)
        if (length(line) > 240)
          line = substr(line, 1, 240)
        print line
      }
    ' "$verilog_log" | sort -u
  )
  if [[ -n "${drop_remark_seen_cases["$case_id"]+x}" ]]; then
    return
  fi
  drop_remark_seen_cases["$case_id"]=1
  drop_remark_cases=$((drop_remark_cases + 1))
  if [[ -n "$BMC_DROP_REMARK_CASES_OUT" ]]; then
    printf "%s\t%s\n" "$case_id" "$case_path" >> "$BMC_DROP_REMARK_CASES_OUT"
  fi
}

load_semantic_tag_map

append_bmc_abstraction_provenance() {
  local case_id="$1"
  local case_path="$2"
  local bmc_log="$3"
  if [[ -z "$BMC_ABSTRACTION_PROVENANCE_OUT" || ! -s "$bmc_log" ]]; then
    return
  fi
  while IFS= read -r line; do
    local token=""
    if [[ "$line" == *"BMC_PROVENANCE_LLHD_INTERFACE "* ]]; then
      token="${line#*BMC_PROVENANCE_LLHD_INTERFACE }"
    elif [[ "$line" == *"BMC_PROVENANCE_LLHD_PROCESS "* ]]; then
      token="process ${line#*BMC_PROVENANCE_LLHD_PROCESS }"
    fi
    if [[ -z "$token" ]]; then
      continue
    fi
    printf "%s\t%s\t%s\n" "$case_id" "$case_path" "$token" \
      >> "$BMC_ABSTRACTION_PROVENANCE_OUT"
  done < "$bmc_log"
}

append_bmc_check_attribution() {
  local case_id="$1"
  local case_path="$2"
  local mlir_file="$3"
  if [[ -z "$BMC_CHECK_ATTRIBUTION_OUT" || ! -s "$mlir_file" ]]; then
    return
  fi
  while IFS=$'\t' read -r idx kind snippet; do
    if [[ -z "$idx" || -z "$kind" || -z "$snippet" ]]; then
      continue
    fi
    printf "%s\t%s\t%s\t%s\t%s\n" \
      "$case_id" "$case_path" "$idx" "$kind" "$snippet" \
      >> "$BMC_CHECK_ATTRIBUTION_OUT"
  done < <(
    awk '
      /verif\.(clocked_assert|assert|clocked_assume|assume|clocked_cover|cover)/ {
        idx++
        kind = "verif.unknown"
        if (match($0, /verif\.[A-Za-z_]+/))
          kind = substr($0, RSTART, RLENGTH)
        line = $0
        gsub(/\t/, " ", line)
        sub(/^[[:space:]]+/, "", line)
        gsub(/[[:space:]]+/, " ", line)
        printf "%d\t%s\t%s\n", idx, kind, line
      }
    ' "$mlir_file"
  )
}

detect_top() {
  local file="$1"
  local requested="$2"
  local modules=()
  while IFS= read -r name; do
    modules+=("$name")
  done < <(awk '
    /^[[:space:]]*module[[:space:]]+/ {
      name=$2
      sub(/[^A-Za-z0-9_$].*/, "", name)
      if (name != "") print name
    }' "$file")

  if [[ -n "$requested" ]]; then
    for m in "${modules[@]}"; do
      if [[ "$m" == "$requested" ]]; then
        echo "$requested"
        return
      fi
    done
  fi

  if [[ ${#modules[@]} -eq 1 ]]; then
    echo "${modules[0]}"
    return
  fi

  if [[ -z "$requested" || "$requested" == "top" ]]; then
    if [[ ${#modules[@]} -gt 0 ]]; then
      echo "${modules[0]}"
      return
    fi
  fi

  echo "$requested"
}

for suite in "${suites[@]}"; do
  if [[ ! -d "$suite" ]]; then
    echo "suite not found: $suite" >&2
    continue
  fi
  while IFS= read -r -d '' sv; do
    base="$(basename "$sv" .sv)"
    if [[ -n "$TEST_FILTER" ]] && ! [[ "$base" =~ $TEST_FILTER ]]; then
      skip=$((skip + 1))
      continue
    fi

    log_tag="$base"
    rel_path="${sv#"$VERIF_DIR/"}"
    if [[ "$rel_path" != "$sv" ]]; then
      log_tag="${rel_path%.sv}"
    fi
    log_tag="${log_tag//\//__}"

    total=$((total + 1))

    mlir="$tmpdir/${base}.mlir"
    verilog_log="$tmpdir/${base}.circt-verilog.log"
    bmc_log="$tmpdir/${base}.circt-bmc.log"
    top_for_file="$(detect_top "$sv" "$TOP")"

    cmd=("$CIRCT_VERILOG" --ir-llhd --timescale=1ns/1ns --single-unit \
      -Wno-implicit-conv -Wno-index-oob -Wno-range-oob -Wno-range-width-oob)
    if [[ "$DISABLE_UVM_AUTO_INCLUDE" == "1" ]]; then
      cmd+=("--no-uvm-auto-include")
    fi
    if [[ -n "$CIRCT_VERILOG_ARGS" ]]; then
      read -r -a extra_args <<<"$CIRCT_VERILOG_ARGS"
      cmd+=("${extra_args[@]}")
    fi
    cmd+=("-I" "$(dirname "$sv")")
    if [[ -n "$top_for_file" ]]; then
      cmd+=("--top=$top_for_file")
    fi
    cmd+=("$sv")

    if ! run_limited "${cmd[@]}" > "$mlir" 2> "$verilog_log"; then
      record_drop_remark_case "$base" "$sv" "$verilog_log"
      result="ERROR"
      if is_xfail "$base"; then
        result="XFAIL"
        xfail=$((xfail + 1))
      else
        error=$((error + 1))
      fi
      emit_result_row "$result" "$base" "$sv"
      continue
    fi
    record_drop_remark_case "$base" "$sv" "$verilog_log"

    bmc_args=("-b" "$BOUND" "--ignore-asserts-until=$IGNORE_ASSERTS_UNTIL" \
      "--module" "$top_for_file")
    if [[ "$BMC_SMOKE_ONLY" == "1" ]]; then
      bmc_args+=("--emit-mlir")
    else
      if [[ "$BMC_RUN_SMTLIB" == "1" ]]; then
        bmc_args+=("--run-smtlib" "--z3-path=$Z3_BIN")
      else
        bmc_args+=("--shared-libs=$Z3_LIB")
      fi
    fi
    if [[ "$BMC_SMOKE_ONLY" != "1" && "$BMC_FAIL_ON_VIOLATION" == "1" ]]; then
      bmc_args+=("--fail-on-violation")
    fi
    if [[ "$RISING_CLOCKS_ONLY" == "1" ]]; then
      bmc_args+=("--rising-clocks-only")
    fi
    if [[ "$BMC_ASSUME_KNOWN_INPUTS" == "1" ]]; then
      bmc_args+=("--assume-known-inputs")
    fi
    if [[ "$ALLOW_MULTI_CLOCK" == "1" ]]; then
      bmc_args+=("--allow-multi-clock")
    fi
    if [[ -n "$CIRCT_BMC_ARGS" ]]; then
      read -r -a extra_bmc_args <<<"$CIRCT_BMC_ARGS"
      bmc_args+=("${extra_bmc_args[@]}")
    fi
    append_bmc_check_attribution "$base" "$sv" "$mlir"
    out=""
    if out="$(run_limited "$CIRCT_BMC" "${bmc_args[@]}" "$mlir" 2> "$bmc_log")"; then
      bmc_status=0
    else
      bmc_status=$?
    fi
    append_bmc_abstraction_provenance "$base" "$sv" "$bmc_log"
    if [[ "$NO_PROPERTY_AS_SKIP" == "1" ]] && \
        grep -q "no property provided to check in module" "$bmc_log"; then
      result="SKIP"
      skip=$((skip + 1))
      emit_result_row "$result" "$base" "$sv"
      if [[ -n "$KEEP_LOGS_DIR" ]]; then
        mkdir -p "$KEEP_LOGS_DIR"
        cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
        cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
          2>/dev/null || true
        cp -f "$bmc_log" "$KEEP_LOGS_DIR/${log_tag}.circt-bmc.log" \
          2>/dev/null || true
      fi
      continue
    fi

    if [[ "$BMC_SMOKE_ONLY" == "1" ]]; then
      if [[ "$bmc_status" -eq 124 || "$bmc_status" -eq 137 ]]; then
        result="TIMEOUT"
      elif [[ "$bmc_status" -eq 0 ]]; then
        result="PASS"
      else
        result="ERROR"
      fi
    else
      if [[ "$bmc_status" -eq 124 || "$bmc_status" -eq 137 ]]; then
        result="TIMEOUT"
      elif grep -q "BMC_RESULT=UNKNOWN" <<<"$out"; then
        result="UNKNOWN"
      elif grep -q "BMC_RESULT=UNSAT" <<<"$out"; then
        result="PASS"
      elif grep -q "BMC_RESULT=SAT" <<<"$out"; then
        result="FAIL"
      else
        result="ERROR"
      fi
    fi

    if [[ "$BMC_SMOKE_ONLY" == "1" ]] && is_xfail "$base"; then
      result="XFAIL"
      xfail=$((xfail + 1))
      emit_result_row "$result" "$base" "$sv"
      if [[ -n "$KEEP_LOGS_DIR" ]]; then
        mkdir -p "$KEEP_LOGS_DIR"
        cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
        cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
          2>/dev/null || true
        cp -f "$bmc_log" "$KEEP_LOGS_DIR/${log_tag}.circt-bmc.log" \
          2>/dev/null || true
      fi
      continue
    fi

    if is_xfail "$base"; then
      if [[ "$result" == "PASS" ]]; then
        result="XPASS"
        xpass=$((xpass + 1))
      else
        result="XFAIL"
        xfail=$((xfail + 1))
      fi
    else
      case "$result" in
        PASS) pass=$((pass + 1)) ;;
        FAIL) fail=$((fail + 1)) ;;
        UNKNOWN)
          unknown=$((unknown + 1))
          error=$((error + 1))
          ;;
        TIMEOUT)
          timeout=$((timeout + 1))
          error=$((error + 1))
          ;;
        *) error=$((error + 1)) ;;
      esac
    fi

    emit_result_row "$result" "$base" "$sv"
    if [[ -n "$KEEP_LOGS_DIR" ]]; then
      mkdir -p "$KEEP_LOGS_DIR"
      cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
      cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
        2>/dev/null || true
      cp -f "$bmc_log" "$KEEP_LOGS_DIR/${log_tag}.circt-bmc.log" \
        2>/dev/null || true
    fi
  done < <(find "$suite" -type f -name "*.sv" -print0)
done

sort "$results_tmp" > "$OUT"
if [[ -n "$BMC_ABSTRACTION_PROVENANCE_OUT" && -f "$BMC_ABSTRACTION_PROVENANCE_OUT" ]]; then
  sort -u -o "$BMC_ABSTRACTION_PROVENANCE_OUT" "$BMC_ABSTRACTION_PROVENANCE_OUT"
fi
if [[ -n "$BMC_CHECK_ATTRIBUTION_OUT" && -f "$BMC_CHECK_ATTRIBUTION_OUT" ]]; then
  sort -u -o "$BMC_CHECK_ATTRIBUTION_OUT" "$BMC_CHECK_ATTRIBUTION_OUT"
fi
if [[ -n "$BMC_DROP_REMARK_CASES_OUT" && -f "$BMC_DROP_REMARK_CASES_OUT" ]]; then
  sort -u -o "$BMC_DROP_REMARK_CASES_OUT" "$BMC_DROP_REMARK_CASES_OUT"
fi
if [[ -n "$BMC_DROP_REMARK_REASONS_OUT" && -f "$BMC_DROP_REMARK_REASONS_OUT" ]]; then
  sort -u -o "$BMC_DROP_REMARK_REASONS_OUT" "$BMC_DROP_REMARK_REASONS_OUT"
fi

echo "verilator-verification summary: total=$total pass=$pass fail=$fail xfail=$xfail xpass=$xpass error=$error skip=$skip unknown=$unknown timeout=$timeout"
echo "verilator-verification dropped-syntax summary: drop_remark_cases=$drop_remark_cases pattern='$DROP_REMARK_PATTERN'"
echo "results: $OUT"
if [[ "$FAIL_ON_DROP_REMARKS" == "1" && "$drop_remark_cases" -gt 0 ]]; then
  echo "FAIL_ON_DROP_REMARKS triggered: drop_remark_cases=$drop_remark_cases" >&2
  exit 2
fi
