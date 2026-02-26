// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// Some pipelines preserve unresolved expr clock keys in bmc_reg_clock_sources
// without arg_index, but keep per-register clock names in bmc_reg_clocks.
// VerifToSMT should use those paired names to map the expr key for clocked
// checks.
// CHECK: smt.solver
func.func @bmc_multiclock_check_key_reg_clock_name_fallback() -> i1 {
  %res = verif.bmc bound 1 num_regs 1 initial_values [false] attributes {
    bmc_input_names = ["clk0", "clk1", "r0_state"],
    bmc_clock_sources = [
      {arg_index = 0 : i32, clock_pos = 0 : i32, invert = false},
      {arg_index = 1 : i32, clock_pos = 1 : i32, invert = false}
    ],
    bmc_clock_keys = ["port:clk0", "port:clk1"],
    bmc_reg_clocks = ["clk1"],
    bmc_reg_clock_sources = [
      {invert = false, clock_key = "expr:reg_clk_expr"}
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
  ^bb0(%c0: !seq.clock, %c1: !seq.clock, %s0: i1):
    %t = hw.constant true
    verif.assert %t label "" {bmc.clock_edge = #ltl<clock_edge posedge>, bmc.clock_key = "expr:reg_clk_expr"} : i1
    verif.yield %s0 : i1
  }
  func.return %res : i1
}
