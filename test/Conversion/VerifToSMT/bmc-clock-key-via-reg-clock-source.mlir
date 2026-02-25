// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// LowerToBMC can leave check metadata keyed to a source clock alias only in
// bmc_reg_clock_sources.clock_key (without a matching bmc_reg_clocks name).
// VerifToSMT should still resolve that key to a BMC clock input.
// CHECK: smt.solver
func.func @bmc_clock_key_via_reg_clock_source() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk_i", "sig"],
    bmc_clock_sources = [{arg_index = 0 : i32, clock_pos = 0 : i32, invert = false}],
    bmc_clock_keys = ["port:clk_i"],
    bmc_reg_clock_sources = [{arg_index = 0 : i32, invert = false, clock_key = "port:clk_src_i"}]
  } init {
    %false = hw.constant false
    %clk = seq.to_clock %false
    verif.yield %clk : !seq.clock
  } loop {
  ^bb0(%clk: !seq.clock):
    verif.yield %clk : !seq.clock
  } circuit {
  ^bb0(%clk: !seq.clock, %sig: i1):
    %true = hw.constant true
    verif.assert %true {bmc.clock = "clk_src_i", bmc.clock_key = "port:clk_src_i", bmc.clock_edge = #ltl<clock_edge posedge>} : i1
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
