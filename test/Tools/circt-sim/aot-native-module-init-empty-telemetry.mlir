// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s

// CHECK: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// CHECK: [circt-compile] Native module init modules: 0 emitted / 1 total
// CHECK: [circt-compile] Top native module init skip reasons:
// CHECK: 1x empty

func.func @identity(%arg0: i32) -> i32 {
  return %arg0 : i32
}

hw.module @top() {
  hw.output
}
