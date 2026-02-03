// RUN: not circt-opt %s --convert-verif-to-smt --verify-diagnostics \
// RUN:   -allow-unregistered-dialect 2>&1 | FileCheck %s

// CHECK: clocked property uses a clock that is not a BMC clock input
func.func @bmc_unmapped_clock() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["sig"]
  }
  init {
    verif.yield
  }
  loop {
    verif.yield
  }
  circuit {
  ^bb0(%sig: i1):
    %clocked = ltl.clock %sig, posedge %sig : i1
    verif.assert %clocked : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
