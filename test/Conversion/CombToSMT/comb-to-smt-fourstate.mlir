// RUN: circt-opt %s --convert-comb-to-smt | FileCheck %s
// XFAIL: *

// CHECK-LABEL: func.func @fourstate_and
// CHECK-DAG: %[[A_VAL:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[A_UNK:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[B_VAL:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[B_UNK:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK: %[[VAL_AND:.*]] = smt.bv.and %[[A_VAL]], %[[B_VAL]] : !smt.bv<2>
// CHECK: %[[UNK_ANY:.*]] = smt.bv.or %[[A_UNK]], %[[B_UNK]] : !smt.bv<2>
// CHECK: %[[A_VAL_NOT:.*]] = smt.bv.not %[[A_VAL]] : !smt.bv<2>
// CHECK: %[[A_UNK_NOT:.*]] = smt.bv.not %[[A_UNK]] : !smt.bv<2>
// CHECK: %[[A_KNOWN0:.*]] = smt.bv.and %[[A_UNK_NOT]], %[[A_VAL_NOT]] : !smt.bv<2>
// CHECK: %[[B_VAL_NOT:.*]] = smt.bv.not %[[B_VAL]] : !smt.bv<2>
// CHECK: %[[B_UNK_NOT:.*]] = smt.bv.not %[[B_UNK]] : !smt.bv<2>
// CHECK: %[[B_KNOWN0:.*]] = smt.bv.and %[[B_UNK_NOT]], %[[B_VAL_NOT]] : !smt.bv<2>
// CHECK: %[[KNOWN0:.*]] = smt.bv.or %[[A_KNOWN0]], %[[B_KNOWN0]] : !smt.bv<2>
// CHECK: %[[NO_KNOWN0:.*]] = smt.bv.not %[[KNOWN0]] : !smt.bv<2>
// CHECK: %[[UNK_AND:.*]] = smt.bv.and %[[UNK_ANY]], %[[NO_KNOWN0]] : !smt.bv<2>
// CHECK: %[[OUT_AND:.*]] = smt.bv.concat %[[VAL_AND]], %[[UNK_AND]] : !smt.bv<2>, !smt.bv<2>
// CHECK: return %[[OUT_AND]] : !smt.bv<4>
hw.module @fourstate_and(in %a: !hw.struct<value: i2, unknown: i2>,
                         in %b: !hw.struct<value: i2, unknown: i2>,
                         out out: !hw.struct<value: i2, unknown: i2>) {
  %0 = comb.and %a, %b : !hw.struct<value: i2, unknown: i2>
  hw.output %0 : !hw.struct<value: i2, unknown: i2>
}

// CHECK-LABEL: func.func @fourstate_or
// CHECK-DAG: %[[A_VAL2:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[A_UNK2:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[B_VAL2:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[B_UNK2:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK: %[[VAL_OR:.*]] = smt.bv.or %[[A_VAL2]], %[[B_VAL2]] : !smt.bv<2>
// CHECK: %[[UNK_ANY2:.*]] = smt.bv.or %[[A_UNK2]], %[[B_UNK2]] : !smt.bv<2>
// CHECK: %[[A_UNK2_NOT:.*]] = smt.bv.not %[[A_UNK2]] : !smt.bv<2>
// CHECK: %[[A_KNOWN1:.*]] = smt.bv.and %[[A_UNK2_NOT]], %[[A_VAL2]] : !smt.bv<2>
// CHECK: %[[B_UNK2_NOT:.*]] = smt.bv.not %[[B_UNK2]] : !smt.bv<2>
// CHECK: %[[B_KNOWN1:.*]] = smt.bv.and %[[B_UNK2_NOT]], %[[B_VAL2]] : !smt.bv<2>
// CHECK: %[[KNOWN1:.*]] = smt.bv.or %[[A_KNOWN1]], %[[B_KNOWN1]] : !smt.bv<2>
// CHECK: %[[NO_KNOWN1:.*]] = smt.bv.not %[[KNOWN1]] : !smt.bv<2>
// CHECK: %[[UNK_OR:.*]] = smt.bv.and %[[UNK_ANY2]], %[[NO_KNOWN1]] : !smt.bv<2>
// CHECK: %[[OUT_OR:.*]] = smt.bv.concat %[[VAL_OR]], %[[UNK_OR]] : !smt.bv<2>, !smt.bv<2>
// CHECK: return %[[OUT_OR]] : !smt.bv<4>
hw.module @fourstate_or(in %a: !hw.struct<value: i2, unknown: i2>,
                        in %b: !hw.struct<value: i2, unknown: i2>,
                        out out: !hw.struct<value: i2, unknown: i2>) {
  %0 = comb.or %a, %b : !hw.struct<value: i2, unknown: i2>
  hw.output %0 : !hw.struct<value: i2, unknown: i2>
}

// CHECK-LABEL: func.func @fourstate_xor
// CHECK-DAG: %[[A_VAL3:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[A_UNK3:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[B_VAL3:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[B_UNK3:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK: %[[VAL_XOR:.*]] = smt.bv.xor %[[A_VAL3]], %[[B_VAL3]] : !smt.bv<2>
// CHECK: %[[UNK_XOR:.*]] = smt.bv.or %[[A_UNK3]], %[[B_UNK3]] : !smt.bv<2>
// CHECK: %[[OUT_XOR:.*]] = smt.bv.concat %[[VAL_XOR]], %[[UNK_XOR]] : !smt.bv<2>, !smt.bv<2>
// CHECK: return %[[OUT_XOR]] : !smt.bv<4>
hw.module @fourstate_xor(in %a: !hw.struct<value: i2, unknown: i2>,
                         in %b: !hw.struct<value: i2, unknown: i2>,
                         out out: !hw.struct<value: i2, unknown: i2>) {
  %0 = comb.xor %a, %b : !hw.struct<value: i2, unknown: i2>
  hw.output %0 : !hw.struct<value: i2, unknown: i2>
}

// CHECK-LABEL: func.func @fourstate_ceq
// CHECK-DAG: %[[A_VAL4:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[A_UNK4:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[B_VAL4:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[B_UNK4:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK: %[[VAL_XOR4:.*]] = smt.bv.xor %[[A_VAL4]], %[[B_VAL4]] : !smt.bv<2>
// CHECK: %[[UNK_XOR4:.*]] = smt.bv.xor %[[A_UNK4]], %[[B_UNK4]] : !smt.bv<2>
// CHECK: %[[UNK_ANY4:.*]] = smt.bv.or %[[A_UNK4]], %[[B_UNK4]] : !smt.bv<2>
// CHECK: %[[KNOWN4:.*]] = smt.bv.not %[[UNK_ANY4]] : !smt.bv<2>
// CHECK: %[[MISMATCH_KNOWN:.*]] = smt.bv.and %[[KNOWN4]], %[[VAL_XOR4]] : !smt.bv<2>
// CHECK: %[[MISMATCH4:.*]] = smt.bv.or %[[MISMATCH_KNOWN]], %[[UNK_XOR4]] : !smt.bv<2>
// CHECK: %[[ZERO4:.*]] = smt.bv.constant #smt.bv<0> : !smt.bv<2>
// CHECK: %[[EQ4:.*]] = smt.eq %[[MISMATCH4]], %[[ZERO4]]
// CHECK: %[[ONE1:.*]] = smt.bv.constant #smt.bv<1> : !smt.bv<1>
// CHECK: %[[ZERO1:.*]] = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK: %[[ITE4:.*]] = smt.ite %[[EQ4]], %[[ONE1]], %[[ZERO1]] : !smt.bv<1>
// CHECK: return %[[ITE4]] : !smt.bv<1>
hw.module @fourstate_ceq(in %a: !hw.struct<value: i2, unknown: i2>,
                         in %b: !hw.struct<value: i2, unknown: i2>,
                         out out: i1) {
  %0 = comb.icmp ceq %a, %b : !hw.struct<value: i2, unknown: i2>
  hw.output %0 : i1
}

// CHECK-LABEL: func.func @fourstate_mux
// CHECK-DAG: %[[T_VAL:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[T_UNK:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[F_VAL:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[F_UNK:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK: %[[VAL_OUT:.*]] = smt.ite %{{.*}}, %[[T_VAL]], %[[F_VAL]] : !smt.bv<2>
// CHECK: %[[UNK_OUT:.*]] = smt.ite %{{.*}}, %[[T_UNK]], %[[F_UNK]] : !smt.bv<2>
// CHECK: %[[OUT:.*]] = smt.bv.concat %[[VAL_OUT]], %[[UNK_OUT]] : !smt.bv<2>, !smt.bv<2>
// CHECK: return %[[OUT]] : !smt.bv<4>
hw.module @fourstate_mux(in %cond: i1,
                         in %t: !hw.struct<value: i2, unknown: i2>,
                         in %f: !hw.struct<value: i2, unknown: i2>,
                         out out: !hw.struct<value: i2, unknown: i2>) {
  %0 = comb.mux %cond, %t, %f : !hw.struct<value: i2, unknown: i2>
  hw.output %0 : !hw.struct<value: i2, unknown: i2>
}

// CHECK-LABEL: func.func @fourstate_weq
// CHECK-DAG: %[[LHS_VAL:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[LHS_UNK:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[RHS_VAL:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[RHS_UNK:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK: %[[MASK:.*]] = smt.bv.not %[[RHS_UNK]] : !smt.bv<2>
// CHECK: %[[VAL_XOR:.*]] = smt.bv.xor %[[LHS_VAL]], %[[RHS_VAL]] : !smt.bv<2>
// CHECK: %[[VAL_MASK:.*]] = smt.bv.and %[[MASK]], %[[VAL_XOR]] : !smt.bv<2>
// CHECK: %[[UNK_MASK:.*]] = smt.bv.and %[[MASK]], %[[LHS_UNK]] : !smt.bv<2>
// CHECK: %[[MISMATCH:.*]] = smt.bv.or %[[VAL_MASK]], %[[UNK_MASK]] : !smt.bv<2>
// CHECK: %[[ZERO:.*]] = smt.bv.constant #smt.bv<0> : !smt.bv<2>
// CHECK: %[[EQ:.*]] = smt.eq %[[MISMATCH]], %[[ZERO]]
// CHECK: %[[ONE1:.*]] = smt.bv.constant #smt.bv<1> : !smt.bv<1>
// CHECK: %[[ZERO1:.*]] = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK: %[[OUT_WEQ:.*]] = smt.ite %[[EQ]], %[[ONE1]], %[[ZERO1]] : !smt.bv<1>
// CHECK: return %[[OUT_WEQ]] : !smt.bv<1>
hw.module @fourstate_weq(in %a: !hw.struct<value: i2, unknown: i2>,
                         in %b: !hw.struct<value: i2, unknown: i2>,
                         out out: i1) {
  %0 = comb.icmp weq %a, %b : !hw.struct<value: i2, unknown: i2>
  hw.output %0 : i1
}

// CHECK-LABEL: func.func @fourstate_wne
// CHECK-DAG: %[[MISMATCH2:.*]] = smt.bv.or %{{.*}}, %{{.*}} : !smt.bv<2>
// CHECK: %[[ZERO2:.*]] = smt.bv.constant #smt.bv<0> : !smt.bv<2>
// CHECK: %[[EQ2:.*]] = smt.eq %[[MISMATCH2]], %[[ZERO2]]
// CHECK: %[[NOT2:.*]] = smt.not %[[EQ2]]
// CHECK: %[[ONE2:.*]] = smt.bv.constant #smt.bv<1> : !smt.bv<1>
// CHECK: %[[ZERO2B:.*]] = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK: %[[OUT_WNE:.*]] = smt.ite %[[NOT2]], %[[ONE2]], %[[ZERO2B]] : !smt.bv<1>
// CHECK: return %[[OUT_WNE]] : !smt.bv<1>
hw.module @fourstate_wne(in %a: !hw.struct<value: i2, unknown: i2>,
                         in %b: !hw.struct<value: i2, unknown: i2>,
                         out out: i1) {
  %0 = comb.icmp wne %a, %b : !hw.struct<value: i2, unknown: i2>
  hw.output %0 : i1
}

// CHECK-LABEL: func.func @fourstate_add
// CHECK-DAG: %[[A_VAL:.*]] = smt.bv.extract %{{.*}} from 1 : (!smt.bv<2>) -> !smt.bv<1>
// CHECK-DAG: %[[A_UNK:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<2>) -> !smt.bv<1>
// CHECK-DAG: %[[B_VAL:.*]] = smt.bv.extract %{{.*}} from 1 : (!smt.bv<2>) -> !smt.bv<1>
// CHECK-DAG: %[[B_UNK:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<2>) -> !smt.bv<1>
// CHECK: %[[SUM:.*]] = smt.bv.xor %[[A_VAL]], %{{.*}} : !smt.bv<1>
// CHECK: %[[UNK:.*]] = smt.bv.and %{{.*}}, %{{.*}} : !smt.bv<1>
// CHECK: %[[OUT_ADD:.*]] = smt.bv.concat %[[SUM]], %[[UNK]] : !smt.bv<1>, !smt.bv<1>
// CHECK: return %[[OUT_ADD]] : !smt.bv<2>
hw.module @fourstate_add(in %a: !hw.struct<value: i1, unknown: i1>,
                         in %b: !hw.struct<value: i1, unknown: i1>,
                         out out: !hw.struct<value: i1, unknown: i1>) {
  %0 = comb.add %a, %b : !hw.struct<value: i1, unknown: i1>
  hw.output %0 : !hw.struct<value: i1, unknown: i1>
}

// CHECK-LABEL: func.func @fourstate_sub
// CHECK-DAG: %[[RHS_VAL:.*]] = smt.bv.extract %{{.*}} from 1 : (!smt.bv<2>) -> !smt.bv<1>
// CHECK: %[[RHS_NOT:.*]] = smt.bv.not %[[RHS_VAL]] : !smt.bv<1>
// CHECK: %[[ONE:.*]] = smt.bv.constant #smt.bv<1> : !smt.bv<1>
// CHECK: %[[SUM:.*]] = smt.bv.xor %{{.*}}, %{{.*}} : !smt.bv<1>
// CHECK: %[[OUT_SUB:.*]] = smt.bv.concat %[[SUM]], %{{.*}} : !smt.bv<1>, !smt.bv<1>
// CHECK: return %[[OUT_SUB]] : !smt.bv<2>
hw.module @fourstate_sub(in %a: !hw.struct<value: i1, unknown: i1>,
                         in %b: !hw.struct<value: i1, unknown: i1>,
                         out out: !hw.struct<value: i1, unknown: i1>) {
  %0 = comb.sub %a, %b : !hw.struct<value: i1, unknown: i1>
  hw.output %0 : !hw.struct<value: i1, unknown: i1>
}

// CHECK-LABEL: func.func @fourstate_shl
// CHECK-DAG: %[[LVAL:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[LUNK:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[RVAL:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[RUNK:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK: %[[RUNK_ZERO:.*]] = smt.eq %[[RUNK]], %{{.*}}
// CHECK: %[[RUNK_NONZERO:.*]] = smt.not %[[RUNK_ZERO]]
// CHECK: %[[VAL_SHL:.*]] = smt.bv.shl %[[LVAL]], %[[RVAL]] : !smt.bv<2>
// CHECK: %[[UNK_SHL:.*]] = smt.bv.shl %[[LUNK]], %[[RVAL]] : !smt.bv<2>
// CHECK: %[[VAL_OUT:.*]] = smt.ite %[[RUNK_NONZERO]], %{{.*}}, %[[VAL_SHL]] : !smt.bv<2>
// CHECK: %[[UNK_OUT:.*]] = smt.ite %[[RUNK_NONZERO]], %{{.*}}, %[[UNK_SHL]] : !smt.bv<2>
// CHECK: %[[OUT_SHL:.*]] = smt.bv.concat %[[VAL_OUT]], %[[UNK_OUT]] : !smt.bv<2>, !smt.bv<2>
// CHECK: return %[[OUT_SHL]] : !smt.bv<4>
hw.module @fourstate_shl(in %a: !hw.struct<value: i2, unknown: i2>,
                         in %b: !hw.struct<value: i2, unknown: i2>,
                         out out: !hw.struct<value: i2, unknown: i2>) {
  %0 = comb.shl %a, %b : !hw.struct<value: i2, unknown: i2>
  hw.output %0 : !hw.struct<value: i2, unknown: i2>
}

// CHECK-LABEL: func.func @fourstate_lshr
// CHECK-DAG: %[[LVAL2:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[LUNK2:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[RVAL2:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK: %[[VAL_LSHR:.*]] = smt.bv.lshr %[[LVAL2]], %[[RVAL2]] : !smt.bv<2>
// CHECK: %[[UNK_LSHR:.*]] = smt.bv.lshr %[[LUNK2]], %[[RVAL2]] : !smt.bv<2>
// CHECK: %[[OUT_LSHR:.*]] = smt.bv.concat %{{.*}}, %{{.*}} : !smt.bv<2>, !smt.bv<2>
// CHECK: return %[[OUT_LSHR]] : !smt.bv<4>
hw.module @fourstate_lshr(in %a: !hw.struct<value: i2, unknown: i2>,
                          in %b: !hw.struct<value: i2, unknown: i2>,
                          out out: !hw.struct<value: i2, unknown: i2>) {
  %0 = comb.shru %a, %b : !hw.struct<value: i2, unknown: i2>
  hw.output %0 : !hw.struct<value: i2, unknown: i2>
}

// CHECK-LABEL: func.func @fourstate_ashr
// CHECK-DAG: %[[LVAL3:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[LUNK3:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[RVAL3:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK: %[[VAL_ASHR:.*]] = smt.bv.ashr %[[LVAL3]], %[[RVAL3]] : !smt.bv<2>
// CHECK: %[[UNK_LSHR3:.*]] = smt.bv.lshr %[[LUNK3]], %[[RVAL3]] : !smt.bv<2>
// CHECK: %[[OUT_ASHR:.*]] = smt.bv.concat %{{.*}}, %{{.*}} : !smt.bv<2>, !smt.bv<2>
// CHECK: return %[[OUT_ASHR]] : !smt.bv<4>
hw.module @fourstate_ashr(in %a: !hw.struct<value: i2, unknown: i2>,
                          in %b: !hw.struct<value: i2, unknown: i2>,
                          out out: !hw.struct<value: i2, unknown: i2>) {
  %0 = comb.shrs %a, %b : !hw.struct<value: i2, unknown: i2>
  hw.output %0 : !hw.struct<value: i2, unknown: i2>
}

// CHECK-LABEL: func.func @fourstate_mul
// CHECK-DAG: %[[LVAL:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[RVAL:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK: %[[MUL:.*]] = smt.bv.mul %[[LVAL]], %[[RVAL]] : !smt.bv<2>
// CHECK: %[[OUT_MUL:.*]] = smt.bv.concat %{{.*}}, %{{.*}} : !smt.bv<2>, !smt.bv<2>
// CHECK: return %[[OUT_MUL]] : !smt.bv<4>
hw.module @fourstate_mul(in %a: !hw.struct<value: i2, unknown: i2>,
                         in %b: !hw.struct<value: i2, unknown: i2>,
                         out out: !hw.struct<value: i2, unknown: i2>) {
  %0 = comb.mul %a, %b : !hw.struct<value: i2, unknown: i2>
  hw.output %0 : !hw.struct<value: i2, unknown: i2>
}

// CHECK-LABEL: func.func @fourstate_divu
// CHECK-DAG: %[[LVAL2:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[RVAL2:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK: %[[DIV:.*]] = smt.bv.udiv %[[LVAL2]], %[[RVAL2]] : !smt.bv<2>
// CHECK: %[[OUT_DIV:.*]] = smt.bv.concat %{{.*}}, %{{.*}} : !smt.bv<2>, !smt.bv<2>
// CHECK: return %[[OUT_DIV]] : !smt.bv<4>
hw.module @fourstate_divu(in %a: !hw.struct<value: i2, unknown: i2>,
                          in %b: !hw.struct<value: i2, unknown: i2>,
                          out out: !hw.struct<value: i2, unknown: i2>) {
  %0 = comb.divu %a, %b : !hw.struct<value: i2, unknown: i2>
  hw.output %0 : !hw.struct<value: i2, unknown: i2>
}

// CHECK-LABEL: func.func @fourstate_modu
// CHECK-DAG: %[[LVAL3:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[RVAL3:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK: %[[MOD:.*]] = smt.bv.urem %[[LVAL3]], %[[RVAL3]] : !smt.bv<2>
// CHECK: %[[OUT_MOD:.*]] = smt.bv.concat %{{.*}}, %{{.*}} : !smt.bv<2>, !smt.bv<2>
// CHECK: return %[[OUT_MOD]] : !smt.bv<4>
hw.module @fourstate_modu(in %a: !hw.struct<value: i2, unknown: i2>,
                          in %b: !hw.struct<value: i2, unknown: i2>,
                          out out: !hw.struct<value: i2, unknown: i2>) {
  %0 = comb.modu %a, %b : !hw.struct<value: i2, unknown: i2>
  hw.output %0 : !hw.struct<value: i2, unknown: i2>
}

// CHECK-LABEL: func.func @fourstate_slt
// CHECK-DAG: %[[UNK_OR:.*]] = smt.bv.or %{{.*}}, %{{.*}} : !smt.bv<2>
// CHECK: %[[UNK_ZERO:.*]] = smt.eq %[[UNK_OR]], %{{.*}}
// CHECK: %[[UNK_BOOL:.*]] = smt.not %[[UNK_ZERO]]
// CHECK: %[[CMP:.*]] = smt.bv.cmp slt %{{.*}}, %{{.*}} : !smt.bv<2>
// CHECK: %[[UNK_SYM:.*]] = smt.declare_fun : !smt.bool
// CHECK: %[[ITE:.*]] = smt.ite %[[UNK_BOOL]], %[[UNK_SYM]], %[[CMP]] : !smt.bool
// CHECK: %[[ONE:.*]] = smt.bv.constant #smt.bv<1> : !smt.bv<1>
// CHECK: %[[ZERO:.*]] = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK: %[[OUT:.*]] = smt.ite %[[ITE]], %[[ONE]], %[[ZERO]] : !smt.bv<1>
// CHECK: return %[[OUT]] : !smt.bv<1>
hw.module @fourstate_slt(in %a: !hw.struct<value: i2, unknown: i2>,
                         in %b: !hw.struct<value: i2, unknown: i2>,
                         out out: i1) {
  %0 = comb.icmp slt %a, %b : !hw.struct<value: i2, unknown: i2>
  hw.output %0 : i1
}

// CHECK-LABEL: func.func @fourstate_parity
// CHECK-DAG: %[[P_VAL:.*]] = smt.bv.extract %{{.*}} from 2 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK-DAG: %[[P_UNK:.*]] = smt.bv.extract %{{.*}} from 0 : (!smt.bv<4>) -> !smt.bv<2>
// CHECK: %[[P_BIT0:.*]] = smt.bv.extract %[[P_VAL]] from 0 : (!smt.bv<2>) -> !smt.bv<1>
// CHECK: %[[P_BIT1:.*]] = smt.bv.extract %[[P_VAL]] from 1 : (!smt.bv<2>) -> !smt.bv<1>
// CHECK: %[[PARITY:.*]] = smt.bv.xor %[[P_BIT0]], %[[P_BIT1]] : !smt.bv<1>
// CHECK: %[[U_BIT0:.*]] = smt.bv.extract %[[P_UNK]] from 0 : (!smt.bv<2>) -> !smt.bv<1>
// CHECK: %[[U_BIT1:.*]] = smt.bv.extract %[[P_UNK]] from 1 : (!smt.bv<2>) -> !smt.bv<1>
// CHECK: %[[UNK_OR:.*]] = smt.bv.or %[[U_BIT0]], %[[U_BIT1]] : !smt.bv<1>
// CHECK: %[[ONEP:.*]] = smt.bv.constant #smt.bv<1> : !smt.bv<1>
// CHECK: %[[HAS_UNK:.*]] = smt.eq %[[UNK_OR]], %[[ONEP]]
// CHECK: %[[SYM:.*]] = smt.declare_fun : !smt.bv<1>
// CHECK: %[[OUT:.*]] = smt.ite %[[HAS_UNK]], %[[SYM]], %[[PARITY]] : !smt.bv<1>
// CHECK: return %[[OUT]] : !smt.bv<1>
hw.module @fourstate_parity(in %a: !hw.struct<value: i2, unknown: i2>,
                            out out: i1) {
  %0 = comb.parity %a : !hw.struct<value: i2, unknown: i2>
  hw.output %0 : i1
}
