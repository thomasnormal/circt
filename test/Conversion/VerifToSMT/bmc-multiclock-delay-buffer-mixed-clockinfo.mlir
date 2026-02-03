// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Mixed clock-name (bmc.clock) and clock-value (ltl.clock) contexts for the
// same delay op should not be treated as a conflict.
// CHECK: smt.solver
func.func @delay_multiclock_mixed_clockinfo() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk0", "sig"]
  }
  init {
    %false = hw.constant false
    %clk0 = seq.to_clock %false
    verif.yield %clk0 : !seq.clock
  }
  loop {
  ^bb0(%clk0: !seq.clock):
    %from0 = seq.from_clock %clk0
    %true = hw.constant true
    %nclk0 = comb.xor %from0, %true : i1
    %new0 = seq.to_clock %nclk0
    verif.yield %new0 : !seq.clock
  }
  circuit {
  ^bb0(%clk0: !seq.clock, %sig: i1):
    %seq = ltl.delay %sig, 1, 0 {bmc.clock = "clk0"} : i1
    %clk0_i1 = seq.from_clock %clk0
    %clocked = ltl.clock %seq, posedge %clk0_i1 : !ltl.sequence
    verif.assert %clocked : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
