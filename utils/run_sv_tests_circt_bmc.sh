#!/usr/bin/env bash
set -euo pipefail

SV_TESTS_DIR="${1:-/home/thomas-ahle/sv-tests}"
BOUND="${BOUND:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=utils/formal_toolchain_resolve.sh
source "$SCRIPT_DIR/formal_toolchain_resolve.sh"
COMMON_SH="$SCRIPT_DIR/lib/common.sh"
if [[ -f "$COMMON_SH" ]]; then
  # shellcheck source=utils/lib/common.sh
  source "$COMMON_SH"
fi

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
  if declare -F circt_common_run_with_limits >/dev/null 2>&1; then
    circt_common_run_with_limits "$memory_limit_kb" "$CIRCT_TIMEOUT_SECS" "$@"
    return
  fi
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
  if declare -F circt_common_is_retryable_launch_failure_log >/dev/null 2>&1; then
    circt_common_is_retryable_launch_failure_log "$log_file"
    return
  fi
  if [[ ! -s "$log_file" ]]; then
    return 1
  fi
  grep -Eiq \
    "Text file busy|ETXTBSY|posix_spawn failed|Permission denied|resource temporarily unavailable|Stale file handle|ESTALE|Too many open files|EMFILE|ENFILE|Cannot allocate memory|ENOMEM" \
    "$log_file"
}

compute_retry_backoff_secs() {
  local attempt="$1"
  awk -v attempt="$attempt" -v base="$BMC_LAUNCH_RETRY_BACKOFF_SECS" \
    'BEGIN { printf "%.3f", attempt * base }'
}

classify_retryable_launch_failure_reason() {
  local log_file="$1"
  local exit_code="$2"
  if declare -F circt_common_classify_retryable_launch_failure_reason >/dev/null 2>&1; then
    circt_common_classify_retryable_launch_failure_reason "$log_file" "$exit_code"
    return
  fi
  if [[ -s "$log_file" ]] && grep -Eiq "Text file busy|ETXTBSY" "$log_file"; then
    echo "etxtbsy"
    return 0
  fi
  if [[ -s "$log_file" ]] && grep -Eiq "posix_spawn failed" "$log_file"; then
    echo "posix_spawn_failed"
    return 0
  fi
  if [[ -s "$log_file" ]] && grep -Eiq "Permission denied" "$log_file"; then
    echo "permission_denied"
    return 0
  fi
  if [[ -s "$log_file" ]] && grep -Eiq "resource temporarily unavailable" "$log_file"; then
    echo "resource_temporarily_unavailable"
    return 0
  fi
  if [[ -s "$log_file" ]] && grep -Eiq "Stale file handle|ESTALE" "$log_file"; then
    echo "stale_file_handle"
    return 0
  fi
  if [[ -s "$log_file" ]] && grep -Eiq "Too many open files|EMFILE|ENFILE" "$log_file"; then
    echo "too_many_open_files"
    return 0
  fi
  if [[ -s "$log_file" ]] && grep -Eiq "Cannot allocate memory|ENOMEM" "$log_file"; then
    echo "cannot_allocate_memory"
    return 0
  fi
  echo "retryable_exit_code_${exit_code}"
}

append_bmc_launch_event() {
  local event_kind="$1"
  local case_id="$2"
  local case_path="$3"
  local stage="$4"
  local tool="$5"
  local reason="$6"
  local attempt="$7"
  local delay_secs="$8"
  local exit_code="$9"
  local fallback_tool="${10}"
  if [[ -z "$BMC_LAUNCH_EVENTS_OUT" ]]; then
    return
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$event_kind" "$case_id" "$case_path" "$stage" "$tool" \
    "$reason" "$attempt" "$delay_secs" "$exit_code" "$fallback_tool" \
    >> "$BMC_LAUNCH_EVENTS_OUT"
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
BMC_LAUNCH_EVENTS_OUT="${BMC_LAUNCH_EVENTS_OUT:-}"
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

is_nonneg_int() {
  local value="$1"
  if declare -F circt_common_is_nonneg_int >/dev/null 2>&1; then
    circt_common_is_nonneg_int "$value"
    return
  fi
  [[ "$value" =~ ^[0-9]+$ ]]
}

is_positive_int() {
  local value="$1"
  if declare -F circt_common_is_positive_int >/dev/null 2>&1; then
    circt_common_is_positive_int "$value"
    return
  fi
  [[ "$value" =~ ^[0-9]+$ ]] && [[ "$value" -gt 0 ]]
}

is_nonneg_decimal() {
  local value="$1"
  if declare -F circt_common_is_nonneg_decimal >/dev/null 2>&1; then
    circt_common_is_nonneg_decimal "$value"
    return
  fi
  [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

is_bool_01() {
  local value="$1"
  if declare -F circt_common_is_bool_01 >/dev/null 2>&1; then
    circt_common_is_bool_01 "$value"
    return
  fi
  [[ "$value" == "0" || "$value" == "1" ]]
}

if [[ ! -d "$SV_TESTS_DIR/tests" ]]; then
  echo "sv-tests directory not found: $SV_TESTS_DIR" >&2
  exit 1
fi

if ! is_nonneg_int "$BMC_LAUNCH_RETRY_ATTEMPTS"; then
  echo "invalid BMC_LAUNCH_RETRY_ATTEMPTS: $BMC_LAUNCH_RETRY_ATTEMPTS" >&2
  exit 1
fi
if ! is_nonneg_decimal "$BMC_LAUNCH_RETRY_BACKOFF_SECS"; then
  echo "invalid BMC_LAUNCH_RETRY_BACKOFF_SECS: $BMC_LAUNCH_RETRY_BACKOFF_SECS" >&2
  exit 1
fi
if ! is_bool_01 "$BMC_LAUNCH_COPY_FALLBACK"; then
  echo "invalid BMC_LAUNCH_COPY_FALLBACK: $BMC_LAUNCH_COPY_FALLBACK" >&2
  exit 1
fi
if ! is_positive_int "$CIRCT_MEMORY_LIMIT_GB"; then
  echo "invalid CIRCT_MEMORY_LIMIT_GB: $CIRCT_MEMORY_LIMIT_GB" >&2
  exit 1
fi
if ! is_nonneg_int "$BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_GB"; then
  echo "invalid BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_GB: $BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_GB" >&2
  exit 1
fi
if [[ "$BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_GB" -ne 0 ]] && \
   [[ "$BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_GB" -le "$CIRCT_MEMORY_LIMIT_GB" ]]; then
  echo "BMC_FRONTEND_OOM_RETRY_MEMORY_LIMIT_GB must be greater than CIRCT_MEMORY_LIMIT_GB when enabled" >&2
  exit 1
fi

if [[ -z "$TAG_REGEX" && -z "$TEST_FILTER" ]]; then
  if [[ "$BMC_SMOKE_ONLY" == "1" ]]; then
    TEST_FILTER="."
  else
    echo "must set TAG_REGEX or TEST_FILTER explicitly (no default filter)" >&2
    exit 1
  fi
fi

if [[ "$BMC_RUN_SMTLIB" == "1" && "$BMC_SMOKE_ONLY" != "1" ]]; then
  if [[ -z "$Z3_BIN" ]]; then
    if declare -F circt_common_resolve_tool >/dev/null 2>&1; then
      if circt_common_resolve_tool z3 >/dev/null 2>&1; then
        Z3_BIN="z3"
      elif [[ -x /home/thomas-ahle/z3-install/bin/z3 ]]; then
        Z3_BIN="/home/thomas-ahle/z3-install/bin/z3"
      elif [[ -x /home/thomas-ahle/z3/build/z3 ]]; then
        Z3_BIN="/home/thomas-ahle/z3/build/z3"
      fi
    else
      if command -v z3 >/dev/null 2>&1; then
        Z3_BIN="z3"
      elif [[ -x /home/thomas-ahle/z3-install/bin/z3 ]]; then
        Z3_BIN="/home/thomas-ahle/z3-install/bin/z3"
      elif [[ -x /home/thomas-ahle/z3/build/z3 ]]; then
        Z3_BIN="/home/thomas-ahle/z3/build/z3"
      fi
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
if [[ -n "$BMC_LAUNCH_EVENTS_OUT" ]]; then
  mkdir -p "$(dirname "$BMC_LAUNCH_EVENTS_OUT")"
  : > "$BMC_LAUNCH_EVENTS_OUT"
fi
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
  local digest=""
  if declare -F circt_common_hash_stdin >/dev/null 2>&1; then
    digest="$(printf "%s" "$payload" | circt_common_hash_stdin 2>/dev/null || true)"
    if [[ -n "$digest" ]]; then
      printf "%s\n" "$digest"
      return 0
    fi
  fi
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
  local digest=""
  if declare -F circt_common_sha256_of >/dev/null 2>&1; then
    digest="$(circt_common_sha256_of "$path")"
    case "$digest" in
      "<missing>"|"<unavailable>") ;;
      *)
        printf "%s\n" "$digest"
        return 0
        ;;
    esac
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    cksum "$path" | awk '{print $1}'
  fi
}

