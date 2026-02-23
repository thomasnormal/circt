#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NINJA_JOBS="${NINJA_JOBS:-1}"

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "[wasm-cxx20-warn] build directory not found: $BUILD_DIR" >&2
  exit 1
fi

cache_file="$BUILD_DIR/CMakeCache.txt"
if [[ ! -f "$cache_file" ]]; then
  echo "[wasm-cxx20-warn] missing CMake cache: $cache_file" >&2
  exit 1
fi

cache_std_line="$(grep -E '^CMAKE_CXX_STANDARD:STRING=' "$cache_file" || true)"
if [[ -z "$cache_std_line" ]]; then
  echo "[wasm-cxx20-warn] missing CMAKE_CXX_STANDARD entry in cache" >&2
  exit 1
fi
cache_std_value="${cache_std_line#CMAKE_CXX_STANDARD:STRING=}"
if [[ ! "$cache_std_value" =~ ^[0-9]+$ ]]; then
  echo "[wasm-cxx20-warn] cache C++ standard is non-numeric: $cache_std_value" >&2
  exit 1
fi
if (( cache_std_value < 20 )); then
  echo "[wasm-cxx20-warn] build C++ standard is below 20: $cache_std_value" >&2
  exit 1
fi

if ! command -v ninja >/dev/null 2>&1; then
  echo "[wasm-cxx20-warn] missing ninja in PATH" >&2
  exit 1
fi

rebuild_targets=(
  "tools/circt/tools/circt-tblgen/CMakeFiles/circt-tblgen.dir/FIRRTLAnnotationsGen.cpp.o"
  "tools/circt/tools/circt-tblgen/CMakeFiles/circt-tblgen.dir/FIRRTLIntrinsicsGen.cpp.o"
  "tools/circt/tools/circt-tblgen/CMakeFiles/circt-tblgen.dir/circt-tblgen.cpp.o"
)

warn_patterns=(
  "ambiguous-reversed-operator"
  "-Wambiguous-reversed-operator"
  "c++20-extensions"
  "-Wc++20-extensions"
)

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
log="$tmpdir/cxx20-warning-check.log"

std_flag_is_cpp20_or_newer() {
  local std_flag="$1"
  case "$std_flag" in
    -std=c++2a|-std=gnu++2a|-std=c++2b|-std=gnu++2b|-std=c++2c|-std=gnu++2c)
      return 0
      ;;
  esac
  if [[ "$std_flag" =~ ^-std=(gnu\+\+|c\+\+)([0-9]+)$ ]]; then
    local std_num="${BASH_REMATCH[2]}"
    if (( std_num >= 20 )); then
      return 0
    fi
  fi
  return 1
}

cmd_dump="$tmpdir/circt-tblgen.commands"
ninja -C "$BUILD_DIR" -t commands circt-tblgen >"$cmd_dump"
for src in "FIRRTLAnnotationsGen.cpp" "FIRRTLIntrinsicsGen.cpp" "circt-tblgen.cpp"; do
  cmd_line="$(grep -F -- "$src" "$cmd_dump" | head -n 1 || true)"
  if [[ -z "$cmd_line" ]]; then
    echo "[wasm-cxx20-warn] missing compile command for $src" >&2
    cat "$cmd_dump" >&2
    exit 1
  fi
  if ! grep -Fq -- "emscripten/em++" <<<"$cmd_line"; then
    echo "[wasm-cxx20-warn] compile command for $src is not using emscripten/em++" >&2
    cat "$cmd_dump" >&2
    exit 1
  fi
  std_flag="$(grep -Eo -- '-std=[^[:space:]]+' <<<"$cmd_line" | head -n 1 || true)"
  if [[ -z "$std_flag" ]]; then
    echo "[wasm-cxx20-warn] compile command for $src is missing a -std=... flag" >&2
    cat "$cmd_dump" >&2
    exit 1
  fi
  if ! std_flag_is_cpp20_or_newer "$std_flag"; then
    echo "[wasm-cxx20-warn] compile command for $src does not use C++20-or-newer: $std_flag" >&2
    cat "$cmd_dump" >&2
    exit 1
  fi
done

# Force recompilation of relevant tblgen TUs so warnings are observable.
if ! ninja -C "$BUILD_DIR" -t clean "${rebuild_targets[@]}" >/dev/null 2>&1; then
  echo "[wasm-cxx20-warn] failed to clean rebuild targets" >&2
  exit 1
fi

if ! ninja -C "$BUILD_DIR" -j "$NINJA_JOBS" "${rebuild_targets[@]}" >"$log" 2>&1; then
  echo "[wasm-cxx20-warn] rebuild failed" >&2
  cat "$log" >&2
  exit 1
fi

for pattern in "${warn_patterns[@]}"; do
  if grep -Fq -- "$pattern" "$log"; then
    echo "[wasm-cxx20-warn] found disallowed warning: $pattern" >&2
    cat "$log" >&2
    exit 1
  fi
done

if grep -Eiq -- "(^|[^[:alpha:]])warning:" "$log"; then
  echo "[wasm-cxx20-warn] found unexpected compiler warning in rebuild output" >&2
  cat "$log" >&2
  exit 1
fi

echo "[wasm-cxx20-warn] PASS"
