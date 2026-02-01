// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// CHECK: smt.solver
func.func @bmc_clock_op_icmp_const_posedge() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_clock_keys = ["arg0"]
  }
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
    %clk_i1 = seq.from_clock %clk
    %true = hw.constant true
    %cmp = comb.icmp eq %clk_i1, %true : i1
    %clocked = ltl.clock %sig, posedge %cmp : i1
    verif.assert %clocked : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
