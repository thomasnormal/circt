// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test derived clock mapping via XOR with false (identity).

// CHECK-LABEL: func.func @bmc_derived_clock_xor_false() -> i1
// CHECK: scf.for
// CHECK: smt.ite
func.func @bmc_derived_clock_xor_false() -> i1 {
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
    %false = hw.constant false
    %same = comb.xor %clk_i1, %false : i1
    %eq = comb.icmp eq %same, %drv : i1
    verif.assume %eq : i1
    %clocked = ltl.clock %seq, posedge %drv : !ltl.sequence
    verif.assert %clocked : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
