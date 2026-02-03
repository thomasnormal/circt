// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @test_bmc_no_clock_regs() -> i1
// CHECK: return

func.func @test_bmc_no_clock_regs() -> (i1) {
  %bmc = verif.bmc bound 2 num_regs 1 initial_values [unit]
  init {
    verif.yield
  }
  loop {
    verif.yield
  }
  circuit {
  ^bb0(%state0: i1):
    %true = hw.constant true
    verif.assert %true : i1
    verif.yield %state0 : i1
  }
  func.return %bmc : i1
}
