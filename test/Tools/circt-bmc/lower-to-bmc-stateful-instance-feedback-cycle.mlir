// RUN: circt-opt --lower-to-bmc="top-module=testModule bound=4" %s | FileCheck %s

// This graph-region feedback goes through a stateful instance. LowerToBMC
// should not reject it as a combinational cycle before state lowering.
hw.module @regmod(in %clk : !seq.clock, in %d : i1, out q : i1) {
  %q = seq.compreg %d, %clk : i1
  hw.output %q : i1
}

hw.module @testModule(in %clk : !seq.clock, in %in : i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %next = comb.xor %in, %q : i1
  %q = hw.instance "u" @regmod(clk: %clk: !seq.clock, d: %next: i1) -> (q: i1)
  %true = hw.constant true
  verif.assert %true : i1
  hw.output
}

// CHECK: verif.bmc
