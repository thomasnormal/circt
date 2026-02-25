#!/usr/bin/env bash
# Shared CIRCT toolchain resolution helpers for formal runner scripts.

resolve_default_circt_tool() {
  local tool_name="$1"
  local preferred_dir="${2:-}"
  if [[ -n "$preferred_dir" ]]; then
    printf '%s\n' "${preferred_dir}/${tool_name}"
    return 0
  fi
  if [[ -n "${SCRIPT_DIR:-}" ]]; then
    printf '%s\n' "${SCRIPT_DIR}/../build_test/bin/${tool_name}"
    return 0
  fi
  printf 'build_test/bin/%s\n' "$tool_name"
}

derive_tool_dir_from_verilog() {
  local verilog_tool="$1"
  if [[ "$verilog_tool" == */* ]]; then
    printf '%s\n' "$(dirname "$verilog_tool")"
    return 0
  fi
  if [[ -n "${SCRIPT_DIR:-}" ]]; then
    printf '%s\n' "${SCRIPT_DIR}/../build_test/bin"
    return 0
  fi
  printf '%s\n' 'build_test/bin'
}
