// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=SMTLIB
// RUN: circt-opt %s --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RUNTIME

// SMTLIB: smt.solver
// SMTLIB-DAG: signal_bin_op = "implies"
// SMTLIB-DAG: iff_bin_op = "iff"
// SMTLIB-DAG: witness_name = "event_arm_witness_0_0"
// SMTLIB: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// SMTLIB: smt.not
// SMTLIB: smt.or
// SMTLIB: smt.eq

// RUNTIME: smt.solver
// RUNTIME-DAG: signal_bin_op = "implies"
// RUNTIME-DAG: iff_bin_op = "iff"
// RUNTIME-DAG: witness_name = "event_arm_witness_0_0"
// RUNTIME: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// RUNTIME: smt.not
// RUNTIME: smt.or
// RUNTIME: smt.eq

func.func @bmc_event_arm_witness_implies_iff_structured() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["a", "b", "en"],
    bmc_event_sources = [["signal[0]:both:iff"]],
    bmc_event_source_details = [[{edge = "both", iff_bin_op = "iff", iff_lhs_name = "a", iff_rhs_name = "b", kind = "signal", label = "signal[0]:both:iff", signal_bin_op = "implies", signal_index = 0 : i32, signal_lhs_name = "a", signal_rhs_name = "b"}]]
  }
  init {
    verif.yield
  }
  loop {
    verif.yield
  }
  circuit {
  ^bb0(%a: i1, %b: i1, %en: i1):
    verif.assert %en : i1
    verif.yield
  }
  return
}
