// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=SMTLIB
// RUN: circt-opt %s --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RUNTIME

// SMTLIB: smt.solver
// SMTLIB-DAG: bmc_event_source_details =
// SMTLIB-DAG: signal_lsb = 1 : i32
// SMTLIB-DAG: signal_msb = 1 : i32
// SMTLIB-DAG: signal_reduction = "or"
// SMTLIB-DAG: signal_reduction = "xnor"
// SMTLIB-DAG: iff_reduction = "nor"
// SMTLIB-DAG: witness_name = "event_arm_witness_0_0"
// SMTLIB-DAG: witness_name = "event_arm_witness_0_1"
// SMTLIB-DAG: witness_name = "event_arm_witness_0_2"
// SMTLIB: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// SMTLIB: smt.declare_fun "event_arm_witness_0_1" : !smt.bool
// SMTLIB: smt.declare_fun "event_arm_witness_0_2" : !smt.bool
// SMTLIB: smt.bv.extract
// SMTLIB: smt.distinct
// SMTLIB: smt.not

// RUNTIME: smt.solver
// RUNTIME-DAG: bmc_event_source_details =
// RUNTIME-DAG: signal_lsb = 1 : i32
// RUNTIME-DAG: signal_msb = 1 : i32
// RUNTIME-DAG: signal_reduction = "or"
// RUNTIME-DAG: signal_reduction = "xnor"
// RUNTIME-DAG: iff_reduction = "nor"
// RUNTIME-DAG: witness_name = "event_arm_witness_0_0"
// RUNTIME-DAG: witness_name = "event_arm_witness_0_1"
// RUNTIME-DAG: witness_name = "event_arm_witness_0_2"
// RUNTIME: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// RUNTIME: smt.declare_fun "event_arm_witness_0_1" : !smt.bool
// RUNTIME: smt.declare_fun "event_arm_witness_0_2" : !smt.bool
// RUNTIME: smt.bv.extract
// RUNTIME: smt.distinct
// RUNTIME: smt.not

func.func @bmc_event_arm_witness_structured_metadata() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["bus", "en"],
    bmc_event_sources = [["signal[0]:both:iff", "signal[1]:both", "signal[2]:both:iff"]],
    bmc_event_source_details = [[{edge = "both", iff_name = "en", kind = "signal", label = "signal[0]:both:iff", signal_index = 0 : i32, signal_lsb = 1 : i32, signal_msb = 1 : i32, signal_name = "bus"}, {edge = "both", kind = "signal", label = "signal[1]:both", signal_index = 1 : i32, signal_name = "bus", signal_reduction = "or"}, {edge = "both", iff_name = "bus", iff_reduction = "nor", kind = "signal", label = "signal[2]:both:iff", signal_index = 2 : i32, signal_name = "bus", signal_reduction = "xnor"}]]
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
