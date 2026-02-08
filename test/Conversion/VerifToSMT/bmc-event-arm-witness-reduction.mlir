// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=SMTLIB
// RUN: circt-opt %s --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RUNTIME

// SMTLIB: smt.solver
// SMTLIB-DAG: bmc_event_source_details =
// SMTLIB-DAG: witness_name = "event_arm_witness_0_0"
// SMTLIB-DAG: witness_name = "event_arm_witness_0_1"
// SMTLIB-DAG: witness_name = "event_arm_witness_0_2"
// SMTLIB-DAG: witness_name = "event_arm_witness_0_3"
// SMTLIB: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// SMTLIB: smt.declare_fun "event_arm_witness_0_1" : !smt.bool
// SMTLIB: smt.declare_fun "event_arm_witness_0_2" : !smt.bool
// SMTLIB: smt.declare_fun "event_arm_witness_0_3" : !smt.bool
// SMTLIB-DAG: smt.bv.extract
// SMTLIB: smt.not

// RUNTIME: smt.solver
// RUNTIME-DAG: bmc_event_source_details =
// RUNTIME-DAG: witness_name = "event_arm_witness_0_0"
// RUNTIME-DAG: witness_name = "event_arm_witness_0_1"
// RUNTIME-DAG: witness_name = "event_arm_witness_0_2"
// RUNTIME-DAG: witness_name = "event_arm_witness_0_3"
// RUNTIME: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// RUNTIME: smt.declare_fun "event_arm_witness_0_1" : !smt.bool
// RUNTIME: smt.declare_fun "event_arm_witness_0_2" : !smt.bool
// RUNTIME: smt.declare_fun "event_arm_witness_0_3" : !smt.bool
// RUNTIME-DAG: smt.bv.extract
// RUNTIME: smt.not

func.func @bmc_event_arm_witness_reduction_expr() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["bus", "en"],
    bmc_event_sources = [["signal[0]:both:iff_expr", "signal[1]:both", "signal[2]:both", "signal[3]:both"]],
    bmc_event_source_details = [[{edge = "both", iff_expr = "&bus", kind = "signal", label = "signal[0]:both:iff_expr", signal_expr = "|bus", signal_index = 0 : i32}, {edge = "both", kind = "signal", label = "signal[1]:both", signal_expr = "^bus", signal_index = 1 : i32}, {edge = "both", kind = "signal", label = "signal[2]:both", signal_expr = "~|bus", signal_index = 2 : i32}, {edge = "both", kind = "signal", label = "signal[3]:both", signal_expr = "^~bus", signal_index = 3 : i32}]]
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
