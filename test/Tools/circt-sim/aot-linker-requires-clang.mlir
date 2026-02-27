// RUN: rm -rf %t.bin && mkdir -p %t.bin
// RUN: echo '#!/bin/sh' > %t.bin/g++
// RUN: echo 'out=""' >> %t.bin/g++
// RUN: echo 'while [ $# -gt 0 ]; do' >> %t.bin/g++
// RUN: echo '  if [ "$1" = "-o" ]; then shift; out="$1"; fi' >> %t.bin/g++
// RUN: echo '  shift' >> %t.bin/g++
// RUN: echo 'done' >> %t.bin/g++
// RUN: echo '[ -n "$out" ] && : > "$out"' >> %t.bin/g++
// RUN: echo 'exit 0' >> %t.bin/g++
// RUN: chmod +x %t.bin/g++
// RUN: not env PATH=%t.bin circt-compile %s -o %t.so 2>&1 | FileCheck %s

// CHECK: Error: cannot find 'clang++' or 'clang' in PATH for linking

func.func @add_i32(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  return %c : i32
}
