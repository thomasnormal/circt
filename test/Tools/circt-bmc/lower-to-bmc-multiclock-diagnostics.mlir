// RUN: circt-opt --lower-to-bmc="top-module=testModule bound=1" --split-input-file --verify-diagnostics %s

// Verify lower-to-bmc surfaces which explicit clock inputs were used when
// rejecting multi-clock designs in single-clock mode.

// expected-error @below {{designs with multiple clocks not yet supported (used explicit clocks: clk0, clk1)}}
hw.module @testModule(in %clk0 : !seq.clock, in %clk1 : !seq.clock, in %in : i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %clk0_i1 = seq.from_clock %clk0
  %clk1_i1 = seq.from_clock %clk1
  %seq0 = ltl.delay %in, 0, 0 : i1
  %seq1 = ltl.delay %in, 0, 0 : i1
  %clocked0 = ltl.clock %seq0, posedge %clk0_i1 : !ltl.sequence
  %clocked1 = ltl.clock %seq1, posedge %clk1_i1 : !ltl.sequence
  verif.assert %clocked0 : !ltl.sequence
  verif.assert %clocked1 : !ltl.sequence
  hw.output
}

// -----

// Verify unresolved clock expressions are called out in diagnostics, so users
// can distinguish explicit-domain conflicts from expression-rooting gaps.

// expected-error @below {{designs with multiple clocks not yet supported (used explicit clocks: clk0) (plus unresolved clock expressions)}}
hw.module @testModule(in %clk0 : !seq.clock, in %clk1 : !seq.clock, in %a : i1, in %b : i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %clk0_i1 = seq.from_clock %clk0
  %seq0 = ltl.delay %a, 0, 0 : i1
  %clocked0 = ltl.clock %seq0, posedge %clk0_i1 : !ltl.sequence
  verif.assert %clocked0 : !ltl.sequence
  %expr_clk = comb.and %a, %b : i1
  %seq1 = ltl.delay %b, 0, 0 : i1
  %clocked1 = ltl.clock %seq1, posedge %expr_clk : !ltl.sequence
  verif.assert %clocked1 : !ltl.sequence
  hw.output
}
