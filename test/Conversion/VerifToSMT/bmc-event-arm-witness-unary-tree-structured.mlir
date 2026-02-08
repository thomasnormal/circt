// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=SMTLIB
// RUN: circt-opt %s --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RUNTIME

// SMTLIB: smt.solver
// SMTLIB-DAG: signal_unary_op = "not"
// SMTLIB-DAG: signal_arg_unary_op = "bitwise_not"
// SMTLIB-DAG: iff_unary_op = "not"
// SMTLIB-DAG: iff_arg_bin_op = "ne"
// SMTLIB-DAG: witness_name = "event_arm_witness_0_0"
// SMTLIB: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// SMTLIB: smt.distinct
// SMTLIB: smt.not

// RUNTIME: smt.solver
// RUNTIME-DAG: signal_unary_op = "not"
// RUNTIME-DAG: signal_arg_unary_op = "bitwise_not"
// RUNTIME-DAG: iff_unary_op = "not"
// RUNTIME-DAG: iff_arg_bin_op = "ne"
// RUNTIME-DAG: witness_name = "event_arm_witness_0_0"
// RUNTIME: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// RUNTIME: smt.distinct
// RUNTIME: smt.not

func.func @bmc_event_arm_witness_unary_tree_structured() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["bus", "en"],
    bmc_event_sources = [["signal[0]:both:iff"]],
    bmc_event_source_details = [[{edge = "both", iff_arg_bin_op = "ne", iff_arg_lhs_lsb = 1 : i32, iff_arg_lhs_msb = 1 : i32, iff_arg_lhs_name = "bus", iff_arg_rhs_name = "en", iff_unary_op = "not", kind = "signal", label = "signal[0]:both:iff", signal_arg_arg_lsb = 0 : i32, signal_arg_arg_msb = 0 : i32, signal_arg_arg_name = "bus", signal_arg_unary_op = "bitwise_not", signal_index = 0 : i32, signal_unary_op = "not"}]]
  }
  init {
    verif.yield
  }
  loop {
    verif.yield
  }
  circuit {
  ^bb0(%bus: i2, %en: i1):
    verif.assert %en : i1
    verif.yield
  }
  return
}
