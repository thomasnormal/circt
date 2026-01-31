// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @bmc_derived_clock_from_to_equivalence()
func.func @bmc_derived_clock_from_to_equivalence() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk"]
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
  ^bb0(%clk: !seq.clock):
    %clk_i1 = seq.from_clock %clk
    %derived_clk = seq.to_clock %clk_i1
    %derived_i1 = seq.from_clock %derived_clk
    %true = ltl.boolean_constant true
    %clocked = ltl.clock %true, posedge %derived_i1 : !ltl.property
    verif.assert %clocked : !ltl.property
    verif.yield %clk_i1 : i1
  }
  func.return %bmc : i1
}
