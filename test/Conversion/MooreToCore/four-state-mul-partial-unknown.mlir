// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @FourStateMulPartialUnknown
// NOTE: i4 prints 0b1000 as -8.
// CHECK: hw.aggregate_constant [0 : i4, -8 : i4]
func.func @FourStateMulPartialUnknown() -> !moore.l4 {
  %a = moore.constant b0001 : l4
  %b = moore.constant bX000 : l4
  %x = moore.mul %a, %b : l4
  return %x : !moore.l4
}
