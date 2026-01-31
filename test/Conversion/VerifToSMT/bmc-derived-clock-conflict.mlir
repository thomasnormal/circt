// RUN: not circt-opt %s --convert-verif-to-smt --verify-diagnostics \
// RUN:   -allow-unregistered-dialect 2>&1 | FileCheck %s

// CHECK: derived clock maps to multiple BMC clock inputs
func.func @bmc_derived_clock_conflict() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk0", "clk1", "sig"]
  }
  init {
    %false = hw.constant false
    %clk0 = seq.to_clock %false
    %clk1 = seq.to_clock %false
    verif.yield %clk0, %clk1 : !seq.clock, !seq.clock
  }
  loop {
  ^bb0(%clk0: !seq.clock, %clk1: !seq.clock):
    verif.yield %clk0, %clk1 : !seq.clock, !seq.clock
  }
  circuit {
  ^bb0(%clk0: !seq.clock, %clk1: !seq.clock, %sig: i1):
    %from0 = seq.from_clock %clk0
    %from1 = seq.from_clock %clk1
    %eq0 = comb.icmp eq %from0, %sig : i1
    %eq1 = comb.icmp eq %from1, %sig : i1
    verif.assume %eq0 : i1
    verif.assume %eq1 : i1
    %seq = ltl.delay %sig, 0, 0 : i1
    %clocked = ltl.clock %seq, posedge %sig : !ltl.sequence
    verif.assert %clocked : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
