// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s
// XFAIL: *
// Known conversion bug: ltl.clock with explicit clock signal causes SSA dominance violation

// Shared ltl.past across multiple clock domains is cloned per property.
// CHECK-LABEL: func.func @past_multiclock_conflict
// CHECK: scf.for
// CHECK: func.call @bmc_circuit
// CHECK-LABEL: func.func @bmc_circuit
func.func @past_multiclock_conflict() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values [] attributes {
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
    %from0 = seq.from_clock %clk0
    %from1 = seq.from_clock %clk1
    %true = hw.constant true
    %nclk0 = comb.xor %from0, %true : i1
    %nclk1 = comb.xor %from1, %true : i1
    %new0 = seq.to_clock %nclk0
    %new1 = seq.to_clock %nclk1
    verif.yield %new0, %new1 : !seq.clock, !seq.clock
  }
  circuit {
  ^bb0(%clk0: !seq.clock, %clk1: !seq.clock, %sig: i1):
    %past = ltl.past %sig, 1 : i1
    verif.assert %past {bmc.clock = "clk0"} : i1
    verif.assert %past {bmc.clock = "clk1"} : i1
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
