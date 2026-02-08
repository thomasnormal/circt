// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=SMTLIB
// RUN: circt-opt %s --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RUNTIME

// SMTLIB: smt.solver
// SMTLIB-DAG: signal_expr = "~bus"
// SMTLIB-DAG: iff_expr = "~bus[2]"
// SMTLIB: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// SMTLIB: smt.bv.extract
// SMTLIB: smt.distinct

// RUNTIME: smt.solver
// RUNTIME-DAG: signal_expr = "~bus"
// RUNTIME-DAG: iff_expr = "~bus[2]"
// RUNTIME: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// RUNTIME: smt.bv.extract
// RUNTIME: smt.distinct

func.func @bmc_event_arm_witness_bitwise_not_text() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["bus", "en"],
    bmc_event_sources = [["signal[0]:both:iff_expr"]],
    bmc_event_source_details = [[{edge = "both", iff_expr = "~bus[2]", kind = "signal", label = "signal[0]:both:iff_expr", signal_expr = "~bus", signal_index = 0 : i32}]]
  }
  init {
    verif.yield
  }
  loop {
    verif.yield
  }
  circuit {
  ^bb0(%bus: i4, %en: i1):
    verif.assert %en : i1
    verif.yield
  }
  return
}
