// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=SMTLIB
// RUN: circt-opt %s --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RUNTIME

// SMTLIB: smt.solver
// SMTLIB-DAG: signal_group
// SMTLIB-DAG: signal_lhs_group
// SMTLIB-DAG: iff_group
// SMTLIB-DAG: witness_name = "event_arm_witness_0_0"
// SMTLIB: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// SMTLIB: smt.and

// RUNTIME: smt.solver
// RUNTIME-DAG: signal_group
// RUNTIME-DAG: signal_lhs_group
// RUNTIME-DAG: iff_group
// RUNTIME-DAG: witness_name = "event_arm_witness_0_0"
// RUNTIME: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// RUNTIME: smt.and

func.func @bmc_event_arm_witness_group_structured() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["bus", "en"],
    bmc_event_sources = [["signal[0]:both:iff"]],
    bmc_event_source_details = [[{edge = "both", iff_group, iff_lsb = 1 : i32, iff_msb = 1 : i32, iff_name = "bus", kind = "signal", label = "signal[0]:both:iff", signal_bin_op = "and", signal_group, signal_index = 0 : i32, signal_lhs_group, signal_lhs_lsb = 0 : i32, signal_lhs_msb = 0 : i32, signal_lhs_name = "bus", signal_rhs_name = "en"}]]
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
