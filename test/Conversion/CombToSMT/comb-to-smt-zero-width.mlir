// RUN: circt-opt %s --convert-comb-to-smt | FileCheck %s

// CHECK-LABEL: func.func @zero_width_extract_mux_icmp
// CHECK-NOT: comb.extract
// CHECK-NOT: comb.mux
// CHECK-NOT: comb.icmp
// CHECK: smt.bv.constant #smt.bv<0> : !smt.bv<1>
func.func @zero_width_extract_mux_icmp(%a0: !smt.bv<1>, %c: !smt.bv<1>) -> !smt.bv<1> {
  %x = builtin.unrealized_conversion_cast %a0 : !smt.bv<1> to i1
  %cond = builtin.unrealized_conversion_cast %c : !smt.bv<1> to i1
  %e = comb.extract %x from 0 : (i1) -> i0
  %m = comb.mux %cond, %e, %e : i0
  %cmp = comb.icmp ne %m, %e : i0
  %out = builtin.unrealized_conversion_cast %cmp : i1 to !smt.bv<1>
  return %out : !smt.bv<1>
}
