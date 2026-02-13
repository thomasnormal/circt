#!/usr/bin/env bash
set -euo pipefail

SV_TESTS_DIR="${1:-/home/thomas-ahle/sv-tests}"
BOUND="${BOUND:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=utils/formal_toolchain_resolve.sh
source "$SCRIPT_DIR/formal_toolchain_resolve.sh"

# Memory limit settings to prevent system hangs.
CIRCT_MEMORY_LIMIT_GB="${CIRCT_MEMORY_LIMIT_GB:-20}"
CIRCT_TIMEOUT_SECS="${CIRCT_TIMEOUT_SECS:-300}"
# Optional single-shot retry memory ceiling for frontend OOM/resource-guard
# failures. Zero disables the retry.
BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_GB="${BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_GB:-0}"
CIRCT_MEMORY_LIMIT_KB=$((CIRCT_MEMORY_LIMIT_GB * 1024 * 1024))
BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_KB=$((BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_GB * 1024 * 1024))

# Run a command with explicit memory and timeout limits.
run_limited_with_memory_kb() {
  local memory_limit_kb="$1"
  shift
  (
    ulimit -v "$memory_limit_kb" 2>/dev/null || true
    timeout --signal=KILL "$CIRCT_TIMEOUT_SECS" "$@"
  )
}

# Run with the default suite memory limit.
run_limited() {
  run_limited_with_memory_kb "$CIRCT_MEMORY_LIMIT_KB" "$@"
}

is_retryable_launch_failure_log() {
  local log_file="$1"
  if [[ ! -s "$log_file" ]]; then
    return 1
  fi
  grep -Eq "Text file busy|failed to run command .*: Permission denied" "$log_file"
}

compute_retry_backoff_secs() {
  local attempt="$1"
  awk -v attempt="$attempt" -v base="$BMC_LAUNCH_RETRY_BACKOFF_SECS" \
    'BEGIN { printf "%.3f", attempt * base }'
}
IGNORE_ASSERTS_UNTIL="${IGNORE_ASSERTS_UNTIL:-1}"
RISING_CLOCKS_ONLY="${RISING_CLOCKS_ONLY:-0}"
ALLOW_MULTI_CLOCK="${ALLOW_MULTI_CLOCK:-0}"
FORCE_BMC="${FORCE_BMC:-0}"
Z3_LIB="${Z3_LIB:-/home/thomas-ahle/z3-install/lib64/libz3.so}"
CIRCT_VERILOG="${CIRCT_VERILOG:-$(resolve_default_circt_tool "circt-verilog")}"
CIRCT_TOOL_DIR_DEFAULT="$(derive_tool_dir_from_verilog "$CIRCT_VERILOG")"
CIRCT_BMC="${CIRCT_BMC:-$(resolve_default_circt_tool "circt-bmc" "$CIRCT_TOOL_DIR_DEFAULT")}"
CIRCT_BMC_ARGS="${CIRCT_BMC_ARGS:-}"
BMC_MLIR_CACHE_DIR="${BMC_MLIR_CACHE_DIR:-}"
BMC_SMOKE_ONLY="${BMC_SMOKE_ONLY:-0}"
BMC_RUN_SMTLIB="${BMC_RUN_SMTLIB:-0}"
Z3_BIN="${Z3_BIN:-}"
KEEP_LOGS_DIR="${KEEP_LOGS_DIR:-}"
FAIL_ON_DROP_REMARKS="${FAIL_ON_DROP_REMARKS:-0}"
DROP_REMARK_PATTERN="${DROP_REMARK_PATTERN:-will be dropped during lowering}"
BMC_ABSTRACTION_PROVENANCE_OUT="${BMC_ABSTRACTION_PROVENANCE_OUT:-}"
BMC_CHECK_ATTRIBUTION_OUT="${BMC_CHECK_ATTRIBUTION_OUT:-}"
BMC_DROP_REMARK_CASES_OUT="${BMC_DROP_REMARK_CASES_OUT:-}"
BMC_DROP_REMARK_REASONS_OUT="${BMC_DROP_REMARK_REASONS_OUT:-}"
BMC_TIMEOUT_REASON_CASES_OUT="${BMC_TIMEOUT_REASON_CASES_OUT:-}"
BMC_FRONTEND_ERROR_REASON_CASES_OUT="${BMC_FRONTEND_ERROR_REASON_CASES_OUT:-}"
BMC_SEMANTIC_TAG_MAP_FILE="${BMC_SEMANTIC_TAG_MAP_FILE:-}"
# NOTE: NO_PROPERTY_AS_SKIP defaults to 0 because the "no property provided to check"
# warning is SPURIOUS - it's emitted before LTLToCore and LowerClockedAssertLike passes
# run, which convert verif.clocked_assert (!ltl.property type) to verif.assert (i1 type).
# After these passes complete, the actual assertions are present and checked correctly.
# Setting this to 1 would cause false SKIP results (e.g., 9/26 instead of 23/26 pass rate).
NO_PROPERTY_AS_SKIP="${NO_PROPERTY_AS_SKIP:-0}"
TAG_REGEX="${TAG_REGEX:-}"
TEST_FILTER="${TEST_FILTER:-}"
OUT="${OUT:-$PWD/sv-tests-bmc-results.txt}"
mkdir -p "$(dirname "$OUT")" 2>/dev/null || true
DISABLE_UVM_AUTO_INCLUDE="${DISABLE_UVM_AUTO_INCLUDE:-1}"
CIRCT_VERILOG_ARGS="${CIRCT_VERILOG_ARGS:-}"
BMC_LAUNCH_RETRY_ATTEMPTS="${BMC_LAUNCH_RETRY_ATTEMPTS:-4}"
BMC_LAUNCH_RETRY_BACKOFF_SECS="${BMC_LAUNCH_RETRY_BACKOFF_SECS:-0.2}"
BMC_LAUNCH_COPY_FALLBACK="${BMC_LAUNCH_COPY_FALLBACK:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECT_FILE="${EXPECT_FILE:-$SCRIPT_DIR/sv-tests-bmc-expect.txt}"
UVM_TAG_REGEX="${UVM_TAG_REGEX:-(^| )uvm( |$)}"
INCLUDE_UVM_TAGS="${INCLUDE_UVM_TAGS:-0}"
TAG_REGEX_EFFECTIVE="$TAG_REGEX"
if [[ "$INCLUDE_UVM_TAGS" == "1" ]]; then
  if [[ -n "$TAG_REGEX_EFFECTIVE" ]]; then
    TAG_REGEX_EFFECTIVE="($TAG_REGEX_EFFECTIVE)|$UVM_TAG_REGEX"
  else
    TAG_REGEX_EFFECTIVE="$UVM_TAG_REGEX"
  fi
fi

resolve_default_uvm_path() {
  local candidate
  for candidate in \
    "$SCRIPT_DIR/../lib/Runtime/uvm" \
    "$SCRIPT_DIR/../lib/Runtime/uvm-core/src" \
    "$SCRIPT_DIR/../lib/Runtime/uvm-core" \
    "/home/thomas-ahle/uvm-core/src"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "$SCRIPT_DIR/../lib/Runtime/uvm"
}

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

UVM_PATH="${UVM_PATH:-$(resolve_default_uvm_path)}"

if [[ ! -d "$SV_TESTS_DIR/tests" ]]; then
  echo "sv-tests directory not found: $SV_TESTS_DIR" >&2
  exit 1
fi

if ! [[ "$BMC_LAUNCH_RETRY_ATTEMPTS" =~ ^[0-9]+$ ]]; then
  echo "invalid BMC_LAUNCH_RETRY_ATTEMPTS: $BMC_LAUNCH_RETRY_ATTEMPTS" >&2
  exit 1
fi
if ! [[ "$BMC_LAUNCH_RETRY_BACKOFF_SECS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "invalid BMC_LAUNCH_RETRY_BACKOFF_SECS: $BMC_LAUNCH_RETRY_BACKOFF_SECS" >&2
  exit 1
fi
if [[ "$BMC_LAUNCH_COPY_FALLBACK" != "0" && "$BMC_LAUNCH_COPY_FALLBACK" != "1" ]]; then
  echo "invalid BMC_LAUNCH_COPY_FALLBACK: $BMC_LAUNCH_COPY_FALLBACK" >&2
  exit 1
fi
if ! [[ "$CIRCT_MEMORY_LIMIT_GB" =~ ^[0-9]+$ ]] || [[ "$CIRCT_MEMORY_LIMIT_GB" -le 0 ]]; then
  echo "invalid CIRCT_MEMORY_LIMIT_GB: $CIRCT_MEMORY_LIMIT_GB" >&2
  exit 1
fi
if ! [[ "$BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_GB" =~ ^[0-9]+$ ]]; then
  echo "invalid BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_GB: $BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_GB" >&2
  exit 1
fi
if [[ "$BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_GB" -ne 0 ]] && \
   [[ "$BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_GB" -le "$CIRCT_MEMORY_LIMIT_GB" ]]; then
  echo "BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_GB must be greater than CIRCT_MEMORY_LIMIT_GB when enabled" >&2
  exit 1
fi

if [[ -z "$TAG_REGEX" && -z "$TEST_FILTER" ]]; then
  echo "must set TAG_REGEX or TEST_FILTER explicitly (no default filter)" >&2
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

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

results_tmp="$tmpdir/results.txt"
touch "$results_tmp"
if [[ -n "$BMC_TIMEOUT_REASON_CASES_OUT" ]]; then
  mkdir -p "$(dirname "$BMC_TIMEOUT_REASON_CASES_OUT")"
  : > "$BMC_TIMEOUT_REASON_CASES_OUT"
fi
if [[ -n "$BMC_FRONTEND_ERROR_REASON_CASES_OUT" ]]; then
  mkdir -p "$(dirname "$BMC_FRONTEND_ERROR_REASON_CASES_OUT")"
  : > "$BMC_FRONTEND_ERROR_REASON_CASES_OUT"
fi

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
cache_hits=0
cache_misses=0
cache_stores=0

declare -A expect_mode
declare -A semantic_tags_by_case
declare -A drop_remark_seen_cases
declare -A drop_remark_seen_case_reasons
declare -A timeout_reason_seen_case_reasons
declare -A frontend_error_reason_seen_case_reasons

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
    printf "%s\t%s\t%s\tsv-tests\tBMC\tsemantic_buckets=%s\n" \
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

record_timeout_reason_case() {
  local case_id="$1"
  local case_path="$2"
  local reason="$3"
  if [[ -z "$BMC_TIMEOUT_REASON_CASES_OUT" || -z "$reason" ]]; then
    return
  fi
  local reason_key="${case_id}|${reason}"
  if [[ -n "${timeout_reason_seen_case_reasons["$reason_key"]+x}" ]]; then
    return
  fi
  timeout_reason_seen_case_reasons["$reason_key"]=1
  mkdir -p "$(dirname "$BMC_TIMEOUT_REASON_CASES_OUT")"
  printf "%s\t%s\t%s\n" "$case_id" "$case_path" "$reason" >> "$BMC_TIMEOUT_REASON_CASES_OUT"
}

classify_frontend_error_reason() {
  local status="$1"
  local verilog_log="$2"
  if [[ "$status" -eq 124 || "$status" -eq 137 ]]; then
    printf '%s\n' "frontend_command_timeout"
    return
  fi
  if [[ -s "$verilog_log" ]]; then
    if grep -Fq "resource guard triggered: RSS" "$verilog_log"; then
      printf '%s\n' "frontend_resource_guard_rss"
      return
    fi
    if grep -Eq "LLVM ERROR: out of memory|Allocation failed|out of memory" "$verilog_log"; then
      printf '%s\n' "frontend_out_of_memory"
      return
    fi
    if [[ "$status" -eq 126 ]] && grep -Fq "Text file busy" "$verilog_log"; then
      printf '%s\n' "frontend_command_launch_text_file_busy"
      return
    fi
    if [[ "$status" -eq 126 ]] && \
        grep -Eq "failed to run command .*: Permission denied" "$verilog_log"; then
      printf '%s\n' "frontend_command_launch_permission_denied"
      return
    fi
  fi
  printf 'frontend_command_exit_%s\n' "$status"
}

record_frontend_error_reason_case() {
  local case_id="$1"
  local case_path="$2"
  local reason="$3"
  if [[ -z "$BMC_FRONTEND_ERROR_REASON_CASES_OUT" || -z "$reason" ]]; then
    return
  fi
  local reason_key="${case_id}|${reason}"
  if [[ -n "${frontend_error_reason_seen_case_reasons["$reason_key"]+x}" ]]; then
    return
  fi
  frontend_error_reason_seen_case_reasons["$reason_key"]=1
  mkdir -p "$(dirname "$BMC_FRONTEND_ERROR_REASON_CASES_OUT")"
  printf "%s\t%s\t%s\n" "$case_id" "$case_path" "$reason" \
    >> "$BMC_FRONTEND_ERROR_REASON_CASES_OUT"
}

load_semantic_tag_map
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

hash_key() {
  local payload="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    printf "%s" "$payload" | sha256sum | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    printf "%s" "$payload" | shasum -a 256 | awk '{print $1}'
  else
    printf "%s" "$payload" | cksum | awk '{print $1}'
  fi
}

hash_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    cksum "$path" | awk '{print $1}'
  fi
}

while IFS= read -r -d '' sv; do
  tags="$(read_meta tags "$sv")"
  if [[ -z "$tags" ]]; then
    skip=$((skip + 1))
    continue
  fi
  if [[ -n "$TAG_REGEX_EFFECTIVE" ]] && ! [[ "$tags" =~ $TAG_REGEX_EFFECTIVE ]]; then
    skip=$((skip + 1))
    continue
  fi

  base="$(basename "$sv" .sv)"
  if [[ -n "$TEST_FILTER" ]] && ! [[ "$base" =~ $TEST_FILTER ]]; then
    skip=$((skip + 1))
    continue
  fi

  total=$((total + 1))

  type="$(read_meta type "$sv")"
  run_bmc=1
  if [[ "$type" =~ [Pp]arsing ]]; then
    run_bmc=0
    if [[ "$FORCE_BMC" == "1" ]]; then
      run_bmc=1
    fi
  fi
  use_uvm=0
  if [[ "$tags" =~ $UVM_TAG_REGEX ]]; then
    use_uvm=1
  fi

  should_fail="$(read_meta should_fail "$sv")"
  should_fail_because="$(read_meta should_fail_because "$sv")"
  if [[ -n "$should_fail_because" ]]; then
    should_fail="1"
  fi
  force_xfail=0
  expect="${expect_mode[$base]-}"
  case "$expect" in
    skip)
      skip=$((skip + 1))
      continue
      ;;
    compile-only|parse-only)
      run_bmc=0
      ;;
    xfail)
      force_xfail=1
      should_fail="1"
      ;;
  esac

  # sv-tests uses `:should_fail_because:` both for tests that should fail to
  # *compile* (parsing/compilation negative tests) and for tests that should
  # *fail at runtime* (simulation negative tests, e.g. an assertion violation).
  #
  # For formal BMC, the latter category should be treated as PASS when a
  # counterexample exists (SAT for asserts). This avoids incorrectly reporting
  # such tests as XFAIL.
  expect_compile_fail=0
  expect_bmc_violation=0
  if [[ "$should_fail" == "1" ]]; then
    if [[ "$type" =~ [Ss]imulation ]]; then
      expect_bmc_violation=1
    else
      expect_compile_fail=1
    fi
  fi

  files_line="$(read_meta files "$sv")"
  incdirs_line="$(read_meta incdirs "$sv")"
  defines_line="$(read_meta defines "$sv")"
  top_module="$(read_meta top_module "$sv")"
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
  bmc_log="$tmpdir/${base}.circt-bmc.log"

  if [[ "$use_uvm" == "1" && ! -d "$UVM_PATH" ]]; then
    printf "ERROR\t%s\t%s (UVM path not found: %s)\n" \
      "$base" "$sv" "$UVM_PATH" >> "$results_tmp"
    error=$((error + 1))
    continue
  fi

  cmd=("$CIRCT_VERILOG" --ir-llhd --timescale=1ns/1ns --single-unit \
    -Wno-implicit-conv -Wno-index-oob -Wno-range-oob -Wno-range-width-oob)
  if [[ "$DISABLE_UVM_AUTO_INCLUDE" == "1" && "$use_uvm" == "0" ]]; then
    cmd+=("--no-uvm-auto-include")
  fi
  if [[ "$use_uvm" == "1" ]]; then
    cmd+=("--uvm-path=$UVM_PATH")
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

  cache_hit=0
  cache_file=""
  if [[ -n "$BMC_MLIR_CACHE_DIR" ]]; then
    mkdir -p "$BMC_MLIR_CACHE_DIR"
    cache_payload="circt_verilog=${CIRCT_VERILOG}
circt_verilog_args=${CIRCT_VERILOG_ARGS}
disable_uvm_auto_include=${DISABLE_UVM_AUTO_INCLUDE}
use_uvm=${use_uvm}
uvm_path=${UVM_PATH}
top_module=${top_module}
"
    for inc in "${incdirs[@]}"; do
      cache_payload+="incdir=${inc}
"
    done
    for def in "${defines[@]}"; do
      cache_payload+="define=${def}
"
    done
    for f in "${files[@]}"; do
      cache_payload+="file=${f} hash=$(hash_file "$f")
"
    done
    cache_key="$(hash_key "$cache_payload")"
    cache_file="$BMC_MLIR_CACHE_DIR/${cache_key}.mlir"
    if [[ -s "$cache_file" ]]; then
      cp -f "$cache_file" "$mlir"
      : > "$verilog_log"
      echo "[run_sv_tests_circt_bmc] cache-hit key=$cache_key case=$base" >> "$verilog_log"
      cache_hit=1
      cache_hits=$((cache_hits + 1))
    else
      cache_misses=$((cache_misses + 1))
    fi
  fi

  frontend_timeout_reason=""
  bmc_timeout_reason=""
  if [[ "$cache_hit" != "1" ]]; then
    : > "$verilog_log"
    launch_attempt=0
    launch_copy_fallback_used=0
    frontend_memory_retry_used=0
    frontend_memory_limit_kb="$CIRCT_MEMORY_LIMIT_KB"
    frontend_error_reason=""
    while true; do
      if run_limited_with_memory_kb "$frontend_memory_limit_kb" "${cmd[@]}" > "$mlir" 2>> "$verilog_log"; then
        verilog_status=0
      else
        verilog_status=$?
      fi
      if [[ "$verilog_status" -eq 0 ]]; then
        break
      fi
      if [[ "$verilog_status" -eq 126 ]] && \
          is_retryable_launch_failure_log "$verilog_log" && \
          [[ "$launch_attempt" -lt "$BMC_LAUNCH_RETRY_ATTEMPTS" ]]; then
        launch_attempt=$((launch_attempt + 1))
        retry_delay_secs="$(compute_retry_backoff_secs "$launch_attempt")"
        {
          printf '[run_sv_tests_circt_bmc] frontend launch retry attempt=%s delay_secs=%s\n' \
            "$launch_attempt" "$retry_delay_secs"
        } >> "$verilog_log"
        sleep "$retry_delay_secs"
        continue
      fi
      if [[ "$verilog_status" -eq 126 && "$BMC_LAUNCH_COPY_FALLBACK" == "1" && \
            "$launch_copy_fallback_used" -eq 0 ]] && \
          is_retryable_launch_failure_log "$verilog_log"; then
        fallback_verilog="$tmpdir/${base}.circt-verilog-launch-fallback"
        if cp -f "$CIRCT_VERILOG" "$fallback_verilog" 2>> "$verilog_log"; then
          chmod +x "$fallback_verilog" 2>> "$verilog_log" || true
          cmd[0]="$fallback_verilog"
          launch_copy_fallback_used=1
          launch_attempt=0
          {
            printf '[run_sv_tests_circt_bmc] frontend launch fallback copy=%s\n' \
              "$fallback_verilog"
          } >> "$verilog_log"
          continue
        else
          {
            printf '[run_sv_tests_circt_bmc] frontend launch fallback copy failed\n'
          } >> "$verilog_log"
        fi
      fi
      if [[ "$verilog_status" -ne 0 && \
            "$frontend_memory_retry_used" -eq 0 && \
            "$BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_KB" -gt "$frontend_memory_limit_kb" ]]; then
        frontend_error_reason="$(classify_frontend_error_reason "$verilog_status" "$verilog_log")"
        if [[ "$frontend_error_reason" == "frontend_out_of_memory" || \
              "$frontend_error_reason" == "frontend_resource_guard_rss" ]]; then
          {
            printf '[run_sv_tests_circt_bmc] frontend memory retry reason=%s from_kb=%s to_kb=%s\n' \
              "$frontend_error_reason" "$frontend_memory_limit_kb" "$BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_KB"
          } >> "$verilog_log"
          frontend_memory_retry_used=1
          frontend_memory_limit_kb="$BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_KB"
          continue
        fi
      fi
      break
    done
    if [[ "$verilog_status" -ne 0 ]]; then
      record_drop_remark_case "$base" "$sv" "$verilog_log"
      if [[ "$force_xfail" == "1" ]]; then
        result="XFAIL"
        xfail=$((xfail + 1))
      elif [[ "$verilog_status" -eq 124 || "$verilog_status" -eq 137 ]]; then
        # Classify frontend timeouts explicitly so summary timeout/error counters
        # reflect performance regressions instead of generic command failures.
        result="TIMEOUT"
        frontend_timeout_reason="frontend_command_timeout"
        timeout=$((timeout + 1))
        error=$((error + 1))
      # Treat expected compile failures as PASS for negative compilation/parsing
      # tests. Simulation-negative tests are expected to compile and are handled
      # via SAT/UNSAT classification below.
      elif [[ "$expect_compile_fail" == "1" ]]; then
        result="PASS"
        pass=$((pass + 1))
      else
        result="ERROR"
        frontend_error_reason="$(classify_frontend_error_reason "$verilog_status" "$verilog_log")"
        error=$((error + 1))
      fi
      emit_result_row "$result" "$base" "$sv"
      if [[ "$result" == "TIMEOUT" ]]; then
        record_timeout_reason_case "$base" "$sv" "$frontend_timeout_reason"
      elif [[ "$result" == "ERROR" ]]; then
        record_frontend_error_reason_case "$base" "$sv" "$frontend_error_reason"
      fi
      if [[ -n "$KEEP_LOGS_DIR" ]]; then
        mkdir -p "$KEEP_LOGS_DIR"
        cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
        cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
          2>/dev/null || true
      fi
      continue
    fi
    record_drop_remark_case "$base" "$sv" "$verilog_log"
    if [[ -n "$cache_file" && -s "$mlir" ]]; then
      cache_tmp="$cache_file.tmp.$$.$RANDOM"
      cp -f "$mlir" "$cache_tmp" 2>/dev/null || true
      mv -f "$cache_tmp" "$cache_file" 2>/dev/null || true
      cache_stores=$((cache_stores + 1))
    fi
  fi

  if [[ "$run_bmc" == "0" ]]; then
    if [[ "$force_xfail" == "1" ]]; then
      result="XPASS"
      xpass=$((xpass + 1))
    elif [[ "$expect_compile_fail" == "1" ]]; then
      result="FAIL"
      fail=$((fail + 1))
    else
      result="PASS"
      pass=$((pass + 1))
    fi
    emit_result_row "$result" "$base" "$sv"
    if [[ -n "$KEEP_LOGS_DIR" ]]; then
      mkdir -p "$KEEP_LOGS_DIR"
      cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
      cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
        2>/dev/null || true
    fi
    continue
  fi

  # Determine how to interpret SAT/UNSAT for this test.
  #
  # By convention, `circt-bmc` reports SAT when it finds an "interesting"
  # condition:
  # - for asserts: a violation exists
  # - for covers: a witness exists
  #
  # For cover-only tests, SAT is therefore a PASS, while UNSAT is a FAIL.
  check_mode="assert"
  if grep -Fq "verif.cover" "$mlir" && ! grep -Fq "verif.assert" "$mlir"; then
    check_mode="cover"
  fi
  append_bmc_check_attribution "$base" "$sv" "$mlir"

  bmc_base_args=("-b" "$BOUND" "--ignore-asserts-until=$IGNORE_ASSERTS_UNTIL" \
    "--module" "$top_module")
  if [[ "$RISING_CLOCKS_ONLY" == "1" ]]; then
    bmc_base_args+=("--rising-clocks-only")
  fi
  if [[ "$ALLOW_MULTI_CLOCK" == "1" ]]; then
    bmc_base_args+=("--allow-multi-clock")
  fi
  if [[ -n "$CIRCT_BMC_ARGS" ]]; then
    read -r -a extra_bmc_args <<<"$CIRCT_BMC_ARGS"
    bmc_base_args+=("${extra_bmc_args[@]}")
  fi

  bmc_args=("${bmc_base_args[@]}")
  if [[ "$BMC_SMOKE_ONLY" == "1" ]]; then
    bmc_args+=("--emit-mlir")
  elif [[ "$BMC_RUN_SMTLIB" == "1" ]]; then
    bmc_args+=("--run-smtlib" "--z3-path=$Z3_BIN")
  else
    bmc_args+=("--shared-libs=$Z3_LIB")
  fi
  out=""
  if out="$(run_limited "$CIRCT_BMC" "${bmc_args[@]}" "$mlir" 2> "$bmc_log")"; then
    bmc_status=0
  else
    bmc_status=$?
  fi
  if [[ "$bmc_status" -ne 0 && "$BMC_SMOKE_ONLY" != "1" && "$BMC_RUN_SMTLIB" == "1" ]] && \
      grep -Fq "for-smtlib-export does not support LLVM dialect operations inside verif.bmc regions" "$bmc_log"; then
    echo "BMC_RUN_SMTLIB fallback($base): retrying with --run due unsupported SMT-LIB export op(s)" >&2
    {
      echo "[run_sv_tests_circt_bmc] BMC_RUN_SMTLIB fallback($base): unsupported SMT-LIB export op(s), retrying with --run"
    } >> "$bmc_log"
    bmc_args=("${bmc_base_args[@]}" "--shared-libs=$Z3_LIB")
    if out="$(run_limited "$CIRCT_BMC" "${bmc_args[@]}" "$mlir" 2>> "$bmc_log")"; then
      bmc_status=0
    else
      bmc_status=$?
    fi
  fi
  append_bmc_abstraction_provenance "$base" "$sv" "$bmc_log"
  # NOTE: The "no property provided to check" warning is typically spurious.
  # It appears before LTLToCore and LowerClockedAssertLike passes run, but
  # after these passes, verif.clocked_assert (!ltl.property) becomes
  # verif.assert (i1), which is properly checked. This skip logic is disabled
  # by default (NO_PROPERTY_AS_SKIP=0) to avoid false SKIP results.
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
      bmc_timeout_reason="solver_command_timeout"
    elif [[ "$bmc_status" -eq 0 ]]; then
      result="PASS"
    else
      result="ERROR"
    fi
  else
    if [[ "$bmc_status" -eq 124 || "$bmc_status" -eq 137 ]]; then
      result="TIMEOUT"
      bmc_timeout_reason="solver_command_timeout"
    elif grep -q "BMC_RESULT=UNKNOWN" <<<"$out"; then
      result="UNKNOWN"
    elif [[ "$check_mode" == "cover" ]]; then
      if grep -q "BMC_RESULT=SAT" <<<"$out"; then
        result="PASS"
      elif grep -q "BMC_RESULT=UNSAT" <<<"$out"; then
        result="FAIL"
      else
        result="ERROR"
      fi
    elif [[ "$expect_bmc_violation" == "1" ]]; then
      # Simulation-negative tests are expected to have an assertion violation
      # within the bound (SAT).
      if grep -q "BMC_RESULT=SAT" <<<"$out"; then
        result="PASS"
      elif grep -q "BMC_RESULT=UNSAT" <<<"$out"; then
        result="FAIL"
      else
        result="ERROR"
      fi
    elif grep -q "BMC_RESULT=UNSAT" <<<"$out"; then
      result="PASS"
    elif grep -q "BMC_RESULT=SAT" <<<"$out"; then
      result="FAIL"
    else
      result="ERROR"
    fi
  fi

  if [[ "$BMC_SMOKE_ONLY" == "1" && "$should_fail" == "1" && "$expect_bmc_violation" != "1" ]]; then
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

  if [[ "$force_xfail" == "1" ]]; then
    if [[ "$result" == "PASS" ]]; then
      result="XPASS"
      xpass=$((xpass + 1))
    else
      result="XFAIL"
      xfail=$((xfail + 1))
    fi
  elif [[ "$expect_bmc_violation" == "1" ]]; then
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
  elif [[ "$should_fail" == "1" ]]; then
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
  if [[ "$result" == "TIMEOUT" ]]; then
    record_timeout_reason_case "$base" "$sv" "$bmc_timeout_reason"
  fi
  if [[ -n "$KEEP_LOGS_DIR" ]]; then
    mkdir -p "$KEEP_LOGS_DIR"
    cp -f "$mlir" "$KEEP_LOGS_DIR/${log_tag}.mlir" 2>/dev/null || true
    cp -f "$verilog_log" "$KEEP_LOGS_DIR/${log_tag}.circt-verilog.log" \
      2>/dev/null || true
    cp -f "$bmc_log" "$KEEP_LOGS_DIR/${log_tag}.circt-bmc.log" \
      2>/dev/null || true
  fi
done < <(find "$SV_TESTS_DIR/tests" -type f -name "*.sv" -print0)

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
if [[ -n "$BMC_FRONTEND_ERROR_REASON_CASES_OUT" && -f "$BMC_FRONTEND_ERROR_REASON_CASES_OUT" ]]; then
  sort -u -o "$BMC_FRONTEND_ERROR_REASON_CASES_OUT" "$BMC_FRONTEND_ERROR_REASON_CASES_OUT"
fi

echo "sv-tests SVA summary: total=$total pass=$pass fail=$fail xfail=$xfail xpass=$xpass error=$error skip=$skip unknown=$unknown timeout=$timeout"
echo "sv-tests dropped-syntax summary: drop_remark_cases=$drop_remark_cases pattern='$DROP_REMARK_PATTERN'"
if [[ -n "$BMC_MLIR_CACHE_DIR" ]]; then
  echo "sv-tests frontend cache summary: hits=$cache_hits misses=$cache_misses stores=$cache_stores dir=$BMC_MLIR_CACHE_DIR"
fi
echo "results: $OUT"
if [[ "$FAIL_ON_DROP_REMARKS" == "1" && "$drop_remark_cases" -gt 0 ]]; then
  echo "FAIL_ON_DROP_REMARKS triggered: drop_remark_cases=$drop_remark_cases" >&2
  exit 2
fi
