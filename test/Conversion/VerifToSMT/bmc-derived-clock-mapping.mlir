// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// CHECK: smt.solver
func.func @bmc_derived_clock_mapping() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values []
  init {
    %false = hw.constant false
    %clk = seq.to_clock %false
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    verif.yield %clk : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %sig: i1):
    %from = seq.from_clock %clk
    %eq = comb.icmp eq %from, %sig : i1
    verif.assume %eq : i1
    %seq = ltl.delay %sig, 0, 0 : i1
    %clocked = ltl.clock %seq, posedge %sig : !ltl.sequence
    verif.assert %clocked : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
