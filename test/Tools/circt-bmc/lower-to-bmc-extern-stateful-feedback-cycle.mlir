// RUN: circt-opt --lower-to-bmc="top-module=testModule bound=2" %s | FileCheck %s

// Feedback through extern modules can represent sequential state (e.g.
// black-box primitives). LowerToBMC must treat extern instances as stateful
// for cycle breaking, otherwise this gets rejected as a combinational loop.
hw.module.extern @prim_stateful(in %clk : !seq.clock, in %d : i1, out q : i1)

hw.module @testModule(in %clk : !seq.clock, in %in : i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %d = comb.xor %in, %q : i1
  %q = hw.instance "u" @prim_stateful(clk: %clk: !seq.clock, d: %d: i1) -> (q: i1)
  %true = hw.constant true
  verif.assert %true : i1
  hw.output
}

// CHECK: verif.bmc
