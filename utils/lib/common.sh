#!/usr/bin/env bash
# Shared helpers for runner scripts.

circt_common_pick_time_tool() {
  if [[ -x /usr/bin/time ]]; then
    echo "/usr/bin/time"
    return
  fi
  if command -v gtime >/dev/null 2>&1; then
    command -v gtime
    return
  fi
  echo ""
}

circt_common_sha256_of() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "<missing>"
    return
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
    return
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
    return
  fi
  if command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 "$path" | awk '{print $NF}'
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$path" <<'PY'
import hashlib
import sys

h = hashlib.sha256()
with open(sys.argv[1], 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
        h.update(chunk)
print(h.hexdigest())
PY
    return
  fi
  echo "<unavailable>"
}

circt_common_hash_stdin() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | awk '{print $1}'
    return 0
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 | awk '{print $1}'
    return 0
  fi
  if command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 | awk '{print $NF}'
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import hashlib,sys;print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())'
    return 0
  fi
  return 1
}

circt_common_normalize_path() {
  local path="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath -m "$path"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$path" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
    return
  fi
  echo "$path"
}

circt_common_resolve_tool() {
  local tool="$1"
  if [[ "$tool" == */* ]]; then
    if [[ -x "$tool" ]]; then
      printf '%s\n' "$tool"
      return 0
    fi
    return 1
  fi
  local resolved
  resolved="$(command -v "$tool" 2>/dev/null || true)"
  if [[ -n "$resolved" && -x "$resolved" ]]; then
    printf '%s\n' "$resolved"
    return 0
  fi
  return 1
}

circt_common_run_with_limits() {
  local memory_limit_kb="$1"
  local timeout_secs="$2"
  shift 2
  (
    ulimit -v "$memory_limit_kb" 2>/dev/null || true
    timeout --signal=KILL "$timeout_secs" "$@"
  )
}

circt_common_is_retryable_launch_failure_log() {
  local log_file="$1"
  if [[ ! -s "$log_file" ]]; then
    return 1
  fi
  grep -Eiq \
    "Text file busy|ETXTBSY|posix_spawn failed|Permission denied|resource temporarily unavailable|Stale file handle|ESTALE|Too many open files|EMFILE|ENFILE|Cannot allocate memory|ENOMEM" \
    "$log_file"
}

circt_common_classify_retryable_launch_failure_reason() {
  local log_file="$1"
  local exit_code="$2"
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

circt_common_is_nonneg_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

circt_common_is_positive_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}

circt_common_is_nonneg_decimal() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

circt_common_is_bool_01() {
  [[ "$1" == "0" || "$1" == "1" ]]
}
