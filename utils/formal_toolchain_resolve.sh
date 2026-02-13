#!/usr/bin/env bash
# Shared CIRCT toolchain resolution helpers for formal runner scripts.

resolve_default_circt_tool() {
  local tool_name="$1"
  local preferred_dir="${2:-}"
  local candidate

  if [[ -n "$preferred_dir" ]]; then
    candidate="${preferred_dir}/${tool_name}"
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  for candidate in "build/bin/${tool_name}" "build-test/bin/${tool_name}"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if [[ -n "${SCRIPT_DIR:-}" ]]; then
    for candidate in "${SCRIPT_DIR}/../build/bin/${tool_name}" \
                     "${SCRIPT_DIR}/../build-test/bin/${tool_name}"; do
      if [[ -x "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done
  fi

  if command -v "$tool_name" >/dev/null 2>&1; then
    command -v "$tool_name"
    return 0
  fi

  if [[ -n "$preferred_dir" ]]; then
    printf '%s\n' "${preferred_dir}/${tool_name}"
  else
    printf 'build/bin/%s\n' "$tool_name"
  fi
}

derive_tool_dir_from_verilog() {
  local verilog_tool="$1"
  local resolved=""

  if [[ "$verilog_tool" == */* ]]; then
    printf '%s\n' "$(dirname "$verilog_tool")"
    return 0
  fi

  resolved="$(command -v "$verilog_tool" 2>/dev/null || true)"
  if [[ -n "$resolved" ]]; then
    printf '%s\n' "$(dirname "$resolved")"
    return 0
  fi

  if [[ -n "${SCRIPT_DIR:-}" ]]; then
    if [[ -x "${SCRIPT_DIR}/../build/bin/${verilog_tool}" ]]; then
      printf '%s\n' "${SCRIPT_DIR}/../build/bin"
      return 0
    fi
    if [[ -x "${SCRIPT_DIR}/../build-test/bin/${verilog_tool}" ]]; then
      printf '%s\n' "${SCRIPT_DIR}/../build-test/bin"
      return 0
    fi
  fi

  printf '%s\n' 'build/bin'
}
