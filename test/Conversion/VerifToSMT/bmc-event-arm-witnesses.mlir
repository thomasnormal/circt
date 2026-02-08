// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' \
// RUN:   --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK: smt.solver
// CHECK-DAG: bmc_event_source_details =
// CHECK-DAG: witness_name = "event_arm_witness_0_0"
// CHECK-DAG: witness_name = "event_arm_witness_0_1"
// CHECK: smt.declare_fun "event_arm_witness_0_0" : !smt.bool
// CHECK: smt.declare_fun "event_arm_witness_0_1" : !smt.bool
func.func @bmc_event_arm_witnesses() {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["seq", "sig", "en"],
    bmc_event_sources = [["sequence", "signal[0]:posedge:iff"]],
    bmc_event_source_details = [[{kind = "sequence", label = "sequence", sequence_name = "seq"}, {edge = "posedge", iff_name = "en", kind = "signal", label = "signal[0]:posedge:iff", signal_index = 0 : i32, signal_name = "sig"}]]
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
