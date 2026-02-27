// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// If bmc_clock_sources maps only one clock position, and bmc_reg_clock_sources
// carries an arg_index without a clock_key, resolve that arg to the unique
// remaining BMC clock position. This preserves check metadata keyed to
// bmc.clock="clk_i"/bmc.clock_key="port:clk_i".
// CHECK: smt.solver
func.func @bmc_multiclock_check_name_reg_source_arg_only_fallback() -> i1 {
  %res = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["src_clk", "bmc_clock_1", "clk_i", "sig"],
    bmc_clock_sources = [
      {arg_index = 0 : i32, clock_pos = 0 : i32, invert = false}
    ],
    bmc_clock_keys = ["port:src_clk", "expr:reg_clk_expr"],
    bmc_reg_clock_sources = [
      {arg_index = 2 : i32, invert = false}
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
  ^bb0(%c0: !seq.clock, %c1: !seq.clock, %clk_i: i1, %sig: i1):
    verif.assert %sig {
      bmc.clock = "clk_i",
      bmc.clock_key = "port:clk_i",
      bmc.clock_edge = #ltl<clock_edge posedge>
    } : i1
    verif.yield %sig : i1
  }
  func.return %res : i1
}
