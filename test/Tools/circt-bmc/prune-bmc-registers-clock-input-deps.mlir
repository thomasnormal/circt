// RUN: circt-opt %s --externalize-registers='allow-multi-clock=true' --prune-bmc-registers | FileCheck %s

// Ensure inputs feeding derived i1 clock expressions stay live. These inputs
// may be referenced only through bmc_reg_clock_sources clock_key metadata.

hw.module @top(in %a : i1, in %b : i1, in %d : i1) {
  %clk_i1 = comb.and %a, %b : i1
  %clk = seq.to_clock %clk_i1
  %r = seq.compreg %d, %clk : i1
  verif.assert %r : i1
  hw.output
}

// CHECK: hw.module @top(in %a : i1, in %b : i1, in %d : i1, in %r_state : i1, out r_next : i1) attributes {bmc_reg_clock_sources = [{clock_key = "expr:
// CHECK: %{{.+}} = comb.and %a, %b : i1
// CHECK: %{{.+}} = seq.to_clock %{{.+}}
// CHECK: verif.assert %r_state : i1
// CHECK: hw.output %d : i1
