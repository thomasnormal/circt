// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=SMTLIB
// RUN: circt-opt %s --convert-verif-to-smt \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RUNTIME

// SMTLIB: smt.solver
// SMTLIB-DAG: bmc_event_source_details =
// SMTLIB-DAG: witness_name = "event_arm_witness_0_0"
// SMTLIB-DAG: witness_name = "event_arm_witness_0_1"
// SMTLIB-DAG: witness_name = "event_arm_witness_0_2"
// SMTLIB: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// SMTLIB: smt.declare_fun "event_arm_witness_0_1" : !smt.bool
// SMTLIB: smt.declare_fun "event_arm_witness_0_2" : !smt.bool

// RUNTIME: smt.solver
// RUNTIME-DAG: bmc_event_source_details =
// RUNTIME-DAG: witness_name = "event_arm_witness_0_0"
// RUNTIME-DAG: witness_name = "event_arm_witness_0_1"
// RUNTIME-DAG: witness_name = "event_arm_witness_0_2"
// RUNTIME: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// RUNTIME: smt.declare_fun "event_arm_witness_0_1" : !smt.bool
// RUNTIME: smt.declare_fun "event_arm_witness_0_2" : !smt.bool
// RUNTIME: scf.for
// RUNTIME: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// RUNTIME: smt.declare_fun "event_arm_witness_0_1" : !smt.bool
// RUNTIME: smt.declare_fun "event_arm_witness_0_2" : !smt.bool
func.func @bmc_event_arm_witnesses() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["seq", "sig", "en"],
    bmc_event_sources = [["sequence", "signal[0]:posedge:iff", "signal[1]:both:iff_expr"]],
    bmc_event_source_details = [[{kind = "sequence", label = "sequence", sequence_name = "seq"}, {edge = "posedge", iff_name = "en", kind = "signal", label = "signal[0]:posedge:iff", signal_index = 0 : i32, signal_name = "sig"}, {edge = "both", iff_expr = "en", kind = "signal", label = "signal[1]:both:iff_expr", signal_expr = "sig & seq", signal_index = 1 : i32}]]
  }
  init {
    verif.yield
  }
  loop {
    verif.yield
  }
  circuit {
  ^bb0(%seq: i1, %sig: i1, %en: i1):
    verif.assert %sig : i1
    verif.yield
  }
  return
}
