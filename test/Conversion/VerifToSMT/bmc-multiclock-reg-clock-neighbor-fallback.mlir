// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// Some imported metadata has small unresolved expr clock islands in
// bmc_reg_clock_sources / bmc_reg_clocks without arg_index. If neighboring
// register clocks are consistent, VerifToSMT should recover that clock.
// CHECK: smt.solver
func.func @bmc_multiclock_reg_clock_neighbor_fallback() -> i1 {
  %res = verif.bmc bound 1 num_regs 5 initial_values [false, false, false, false, false] attributes {
    bmc_input_names = ["clk0", "clk1", "r0", "r1", "r2", "r3", "r4"],
    bmc_clock_sources = [
      {arg_index = 0 : i32, clock_pos = 0 : i32, invert = false},
      {arg_index = 1 : i32, clock_pos = 1 : i32, invert = false}
    ],
    bmc_clock_keys = ["port:clk0", "port:clk1"],
    bmc_reg_clocks = ["clk0", "clk0", "expr:hole", "expr:hole", "clk0"],
    bmc_reg_clock_sources = [
      {arg_index = 0 : i32, invert = false, clock_key = "port:clk0"},
      {arg_index = 0 : i32, invert = false, clock_key = "port:clk0"},
      {invert = false, clock_key = "expr:hole"},
      {invert = false, clock_key = "expr:hole"},
      {arg_index = 0 : i32, invert = false, clock_key = "port:clk0"}
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
  ^bb0(%c0: !seq.clock, %c1: !seq.clock, %s0: i1, %s1: i1, %s2: i1, %s3: i1, %s4: i1):
    %p = comb.and %s2, %s3 : i1
    verif.assert %p label "" {bmc.clock_edge = #ltl<clock_edge posedge>, bmc.clock_key = "expr:check"} : i1
    verif.yield %s0, %s1, %s2, %s3, %s4 : i1, i1, i1, i1, i1
  }
  func.return %res : i1
}
