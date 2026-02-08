// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// CHECK: smt.solver
// CHECK-SAME: bmc_event_sources =
// CHECK-SAME: "sequence"
// CHECK-SAME: "signal[0]:posedge:iff"
// CHECK-SAME: "signal[0]:both"
func.func @bmc_mixed_event_sources() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_event_sources = [
      ["sequence", "signal[0]:posedge:iff"],
      ["sequence", "signal[0]:both"]
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
