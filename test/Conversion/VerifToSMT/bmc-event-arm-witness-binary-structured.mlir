// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=SMTLIB
// RUN: circt-opt %s --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RUNTIME

// SMTLIB: smt.solver
// SMTLIB-DAG: signal_bin_op = "and"
// SMTLIB-DAG: iff_bin_op = "ne"
// SMTLIB-DAG: witness_name = "event_arm_witness_0_0"
// SMTLIB: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// SMTLIB: smt.and
// SMTLIB: smt.distinct

// RUNTIME: smt.solver
// RUNTIME-DAG: signal_bin_op = "and"
// RUNTIME-DAG: iff_bin_op = "ne"
// RUNTIME-DAG: witness_name = "event_arm_witness_0_0"
// RUNTIME: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// RUNTIME: smt.and
// RUNTIME: smt.distinct

func.func @bmc_event_arm_witness_binary_structured() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["bus", "en"],
    bmc_event_sources = [["signal[0]:both:iff"]],
    bmc_event_source_details = [[{edge = "both", iff_bin_op = "ne", iff_lhs_lsb = 1 : i32, iff_lhs_msb = 1 : i32, iff_lhs_name = "bus", iff_rhs_name = "en", kind = "signal", label = "signal[0]:both:iff", signal_bin_op = "and", signal_index = 0 : i32, signal_lhs_lsb = 0 : i32, signal_lhs_msb = 0 : i32, signal_lhs_name = "bus", signal_rhs_name = "en"}]]
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
