// RUN: circt-opt --convert-hw-to-smt %s | FileCheck %s

hw.module.extern @ext(in %a : i1, out b : i1, out dbg : !llhd.ref<i1>)

hw.module @top(in %a : i1, out y : i1) {
  %b, %dbg = hw.instance "u" @ext(a: %a: i1) -> (b: i1, dbg: !llhd.ref<i1>)
  hw.output %b : i1
}

// CHECK: func.func private @ext(!smt.bv<1>) -> (!smt.bv<1>, !smt.bv<1>)
// CHECK: func.func @top(%[[IN:.*]]: !smt.bv<1>) -> !smt.bv<1>
// CHECK: %[[RES:.*]]:2 = call @ext(%[[IN]]) : (!smt.bv<1>) -> (!smt.bv<1>, !smt.bv<1>)
// CHECK: return %[[RES]]#0 : !smt.bv<1>
