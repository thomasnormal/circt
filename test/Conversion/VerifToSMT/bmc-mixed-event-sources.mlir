// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// CHECK: smt.solver
// CHECK-DAG: bmc_event_sources =
// CHECK-DAG: "sequence"
// CHECK-DAG: "signal[0]:posedge:iff"
// CHECK-DAG: "signal[0]:both"
// CHECK-DAG: bmc_event_source_details =
// CHECK-DAG: signal_name = "clk"
// CHECK-DAG: iff_name = "en"
func.func @bmc_mixed_event_sources() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_event_sources = [
      ["sequence", "signal[0]:posedge:iff"],
      ["sequence", "signal[0]:both"]
    ],
    bmc_event_source_details = [
      [{kind = "sequence", label = "sequence"}, {edge = "posedge", iff_name = "en", kind = "signal", label = "signal[0]:posedge:iff", signal_index = 0 : i32, signal_name = "clk"}],
      [{kind = "sequence", label = "sequence"}, {edge = "both", kind = "signal", label = "signal[0]:both", signal_index = 0 : i32, signal_name = "clk"}]
    ]
  }
  init {
    verif.yield
  }
  loop {
    verif.yield
  }
  circuit {
    %true = hw.constant true
    verif.assert %true : i1
    verif.yield
  }
  func.return %bmc : i1
}
