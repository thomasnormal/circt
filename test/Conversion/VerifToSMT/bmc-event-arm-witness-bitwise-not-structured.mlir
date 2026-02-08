// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=SMTLIB
// RUN: circt-opt %s --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RUNTIME

// SMTLIB: smt.solver
// SMTLIB-DAG: signal_bitwise_not
// SMTLIB-DAG: iff_bitwise_not
// SMTLIB-DAG: witness_name = "event_arm_witness_0_0"
// SMTLIB: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// SMTLIB: smt.bv.extract
// SMTLIB: smt.distinct

// RUNTIME: smt.solver
// RUNTIME-DAG: signal_bitwise_not
// RUNTIME-DAG: iff_bitwise_not
// RUNTIME-DAG: witness_name = "event_arm_witness_0_0"
// RUNTIME: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// RUNTIME: smt.bv.extract
// RUNTIME: smt.distinct

func.func @bmc_event_arm_witness_bitwise_not_structured() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["bus", "en"],
    bmc_event_sources = [["signal[0]:both:iff"]],
    bmc_event_source_details = [[{edge = "both", iff_bitwise_not, iff_lsb = 2 : i32, iff_msb = 2 : i32, iff_name = "bus", kind = "signal", label = "signal[0]:both:iff", signal_bitwise_not, signal_index = 0 : i32, signal_name = "bus"}]]
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
