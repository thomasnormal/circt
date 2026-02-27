// RUN: circt-opt --convert-hw-to-smt %s | FileCheck %s

// CHECK-LABEL: func.func @zero_bit_used
// CHECK-NOT: hw.constant
// CHECK: arith.constant 0 : i0
// CHECK: comb.mux
// CHECK: comb.icmp ne
hw.module @zero_bit_used(in %sel : i1, in %x : i1, out out : i1) {
  %c0_i0 = hw.constant 0 : i0
  %slice = comb.extract %x from 0 : (i1) -> i0
  %mux = comb.mux %sel, %slice, %c0_i0 : i0
  %cmp = comb.icmp ne %mux, %c0_i0 : i0
  hw.output %cmp : i1
}
