// RUN: circt-opt %s --convert-comb-to-smt | FileCheck %s

// CHECK-LABEL: func @tt
// CHECK: [[BCAST:%.+]] = smt.array.broadcast {{.*}} : !smt.array<[!smt.bv<2> -> !smt.bv<1>]>
// CHECK: [[STORE1:%.+]] = smt.array.store [[BCAST]][{{.*}}], {{.*}} : !smt.array<[!smt.bv<2> -> !smt.bv<1>]>
// CHECK: [[STORE2:%.+]] = smt.array.store [[STORE1]][{{.*}}], {{.*}} : !smt.array<[!smt.bv<2> -> !smt.bv<1>]>
// CHECK: [[IDX:%.+]] = smt.bv.concat {{.*}}, {{.*}} : !smt.bv<1>, !smt.bv<1>
// CHECK: smt.array.select [[STORE2]]
func.func @tt(%a0: !smt.bv<1>, %a1: !smt.bv<1>) {
  %a = builtin.unrealized_conversion_cast %a0 : !smt.bv<1> to i1
  %b = builtin.unrealized_conversion_cast %a1 : !smt.bv<1> to i1
  %0 = comb.truth_table %a, %b -> [false, true, true, false]
  return
}
