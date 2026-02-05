// RUN: circt-opt --lower-to-bmc="top-module=top bound=2" %s | FileCheck %s

// CHECK: verif.bmc
// CHECK: bmc_clock_keys = ["expr:{{[0-9A-Fa-f]+}}"]
// CHECK: ltl.clock{{.*}}bmc.clock_key = "expr:{{[0-9A-Fa-f]+}}"

hw.module @top(in %clk: i1, in %en: i1, in %in: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %gated = comb.and %clk, %en : i1
  %clocked = ltl.clock %in, posedge %gated : i1
  verif.assert %clocked : !ltl.sequence
  hw.output
}
