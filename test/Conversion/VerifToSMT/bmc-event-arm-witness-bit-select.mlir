// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=SMTLIB
// RUN: circt-opt %s --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RUNTIME

// SMTLIB: smt.solver
// SMTLIB-DAG: bmc_event_source_details =
// SMTLIB-DAG: witness_name = "event_arm_witness_0_0"
// SMTLIB: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// SMTLIB: smt.bv.extract

// RUNTIME: smt.solver
// RUNTIME-DAG: bmc_event_source_details =
// RUNTIME-DAG: witness_name = "event_arm_witness_0_0"
// RUNTIME: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// RUNTIME: smt.bv.extract

func.func @bmc_event_arm_witness_bit_select() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["bus", "en"],
    bmc_event_sources = [["signal[0]:both:iff_expr"]],
    bmc_event_source_details = [[{edge = "both", iff_expr = "en", kind = "signal", label = "signal[0]:both:iff_expr", signal_expr = "bus[0]", signal_index = 0 : i32}]]
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
