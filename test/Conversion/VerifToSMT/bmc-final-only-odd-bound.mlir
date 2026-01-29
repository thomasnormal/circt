// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @final_only_odd_bound
// CHECK-NOT: smt.check
// CHECK-LABEL: func.func @bmc_circuit
func.func @final_only_odd_bound() -> i1 {
  %bmc = verif.bmc bound 3 num_regs 0 initial_values []
  init {
    %c0 = hw.constant false
    %clk = seq.to_clock %c0
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    %from = seq.from_clock %clk
    %true = hw.constant true
    %n = comb.xor %from, %true : i1
    %nclk = seq.to_clock %n
    verif.yield %nclk : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %sig: i1):
    %prop = comb.and %sig, %sig : i1
    verif.assert %prop {bmc.final} : i1
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
