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

if ! grep -Fq -- "CMAKE_CXX_STANDARD:STRING=20" "$cache_file"; then
  echo "[wasm-cxx20-warn] build is not configured for C++20 (expected CMAKE_CXX_STANDARD:STRING=20)" >&2
  exit 1
fi

if ! command -v ninja >/dev/null 2>&1; then
  echo "[wasm-cxx20-warn] missing ninja in PATH" >&2
  exit 1
fi

rebuild_targets=(
  "tools/circt/tools/circt-tblgen/CMakeFiles/circt-tblgen.dir/FIRRTLAnnotationsGen.cpp.o"
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

cmd_dump="$tmpdir/circt-tblgen.commands"
ninja -C "$BUILD_DIR" -t commands circt-tblgen >"$cmd_dump"
if ! grep -Fq -- "-std=c++20" "$cmd_dump"; then
  echo "[wasm-cxx20-warn] circt-tblgen compile command is not using -std=c++20" >&2
  cat "$cmd_dump" >&2
  exit 1
fi
for src in "FIRRTLAnnotationsGen.cpp" "circt-tblgen.cpp"; do
  if ! grep -F -- "$src" "$cmd_dump" | grep -Fq -- "-std=c++20"; then
    echo "[wasm-cxx20-warn] compile command for $src is missing -std=c++20" >&2
    cat "$cmd_dump" >&2
    exit 1
  fi
done

# Force recompilation of relevant tblgen TUs so warnings are observable.
ninja -C "$BUILD_DIR" -t clean "${rebuild_targets[@]}" >/dev/null 2>&1 || true

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
