// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Ensure neutral boolean ops and eq-with-const are ignored for clock mapping.

// CHECK-LABEL: func.func @bmc_derived_clock_neutral_ops
// CHECK: func.call @bmc_circuit
// CHECK-SAME: -> ({{.*}}!smt.bool)
func.func @bmc_derived_clock_neutral_ops() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk", "sig"]
  }
  init {
    %false = hw.constant false
    %clk = seq.to_clock %false
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    %from = seq.from_clock %clk
    %true = hw.constant true
    %nclk = comb.xor %from, %true : i1
    %new = seq.to_clock %nclk
    verif.yield %new : !seq.clock
  }
  circuit {
  ^bb0(%clk: i1, %sig: i1):
    %true = hw.constant true
    %false = hw.constant false
    %eq = comb.icmp eq %clk, %true : i1
    %and = comb.and bin %eq, %true : i1
    %or = comb.or bin %and, %false : i1
    %seq = ltl.delay %sig, 0, 0 : i1
    %clocked = ltl.clock %seq, posedge %or : !ltl.sequence
    verif.assert %clocked : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
