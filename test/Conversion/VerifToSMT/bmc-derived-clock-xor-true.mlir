// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test derived clock mapping via XOR with true (XNOR semantics).

// CHECK-LABEL: func.func @bmc_derived_clock_xor_true() -> i1
// CHECK: scf.for
// CHECK: smt.ite
func.func @bmc_derived_clock_xor_true() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk", "drv", "sig"]
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
  ^bb0(%clk: !seq.clock, %drv: i1, %sig: i1):
    %seq = ltl.delay %sig, 1, 0 : i1
    %clk_i1 = seq.from_clock %clk
    %true = hw.constant true
    %xnor = comb.xor %clk_i1, %drv, %true : i1
    verif.assume %xnor : i1
    %clocked = ltl.clock %seq, posedge %drv : !ltl.sequence
    verif.assert %clocked : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
