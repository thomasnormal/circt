// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=SMTLIB
// RUN: circt-opt %s --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RUNTIME

// SMTLIB: smt.solver
// SMTLIB-DAG: signal_bin_op = "eq"
// SMTLIB-DAG: signal_lhs_unary_op = "bitwise_not"
// SMTLIB-DAG: iff_bin_op = "ne"
// SMTLIB-DAG: iff_lhs_unary_op = "bitwise_not"
// SMTLIB-DAG: witness_name = "event_arm_witness_0_0"
// SMTLIB: smt.bv.not
// SMTLIB: smt.eq {{.*}} : !smt.bv<2>
// SMTLIB: smt.distinct {{.*}} : !smt.bv<2>

// RUNTIME: smt.solver
// RUNTIME-DAG: signal_bin_op = "eq"
// RUNTIME-DAG: signal_lhs_unary_op = "bitwise_not"
// RUNTIME-DAG: iff_bin_op = "ne"
// RUNTIME-DAG: iff_lhs_unary_op = "bitwise_not"
// RUNTIME-DAG: witness_name = "event_arm_witness_0_0"
// RUNTIME: smt.bv.not
// RUNTIME: smt.eq {{.*}} : !smt.bv<2>
// RUNTIME: smt.distinct {{.*}} : !smt.bv<2>

func.func @bmc_event_arm_witness_eq_ne_nonleaf_structured() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["bus", "mask", "en"],
    bmc_event_sources = [["signal[0]:both:iff"]],
    bmc_event_source_details = [[{edge = "both", iff_bin_op = "ne", iff_lhs_arg_name = "bus", iff_lhs_unary_op = "bitwise_not", iff_rhs_name = "mask", kind = "signal", label = "signal[0]:both:iff", signal_bin_op = "eq", signal_index = 0 : i32, signal_lhs_arg_name = "bus", signal_lhs_unary_op = "bitwise_not", signal_rhs_name = "mask"}]]
  }
  init {
    verif.yield
  }
  loop {
    verif.yield
  }
  circuit {
  ^bb0(%bus: i2, %mask: i2, %en: i1):
    verif.assert %en : i1
    verif.yield
  }
  return
}
