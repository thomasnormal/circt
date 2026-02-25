// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// Some imported multi-clock designs carry a small number of unresolved
// expression clock keys in bmc_reg_clock_sources / bmc_reg_clocks while the
// rest of registers map cleanly via arg_index. VerifToSMT should not fail this
// shape outright.
// CHECK: smt.solver
func.func @bmc_reg_clock_sources_unresolved_expr_fallback() -> i1 {
  %res = verif.bmc bound 1 num_regs 4 initial_values [false, false, false, false] attributes {
    bmc_input_names = ["clk0", "clk1", "r0_state", "r1_state", "r2_state", "r3_state"],
    bmc_clock_sources = [
      {arg_index = 0 : i32, clock_pos = 0 : i32, invert = false},
      {arg_index = 1 : i32, clock_pos = 1 : i32, invert = false}
    ],
    bmc_clock_keys = ["port:clk0", "port:clk1"],
    bmc_reg_clocks = ["clk0_0", "clk1_0", "clk0_0", "expr:deadbeef"],
    bmc_reg_clock_sources = [
      {arg_index = 0 : i32, invert = false, clock_key = "port:clk0"},
      {arg_index = 1 : i32, invert = false, clock_key = "port:clk1"},
      {arg_index = 0 : i32, invert = false, clock_key = "port:clk0"},
      {invert = false, clock_key = "expr:deadbeef"}
    ]
  } init {
    %f = hw.constant false
    %c0 = seq.to_clock %f
    %c1 = seq.to_clock %f
    verif.yield %c0, %c1 : !seq.clock, !seq.clock
  } loop {
  ^bb0(%c0: !seq.clock, %c1: !seq.clock):
    verif.yield %c0, %c1 : !seq.clock, !seq.clock
  } circuit {
  ^bb0(%c0: !seq.clock, %c1: !seq.clock, %s0: i1, %s1: i1, %s2: i1, %s3: i1):
    %t = hw.constant true
    verif.assert %t : i1
    verif.yield %s0, %s1, %s2, %s3 : i1, i1, i1, i1
  }
  func.return %res : i1
}
