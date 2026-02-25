// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// LowerToBMC can preserve check clock names via bmc_reg_clocks even when the
// BMC clock input name differs. VerifToSMT should resolve those names through
// bmc_reg_clock_sources metadata.
// CHECK: smt.solver
func.func @bmc_clock_name_via_reg_clock_source() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 1 initial_values [unit] attributes {
    bmc_clock_sources = [{arg_index = 0 : i32, clock_pos = 0 : i32, invert = false}],
    bmc_reg_clocks = ["aux_clk"],
    bmc_reg_clock_sources = [{arg_index = 0 : i32, invert = false}]
  }
  init {
    %false = hw.constant false
    %clk = seq.to_clock %false
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    verif.yield %clk : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %state: i1):
    %true = hw.constant true
    verif.assert %true {bmc.clock = "aux_clk", bmc.clock_edge = #ltl<clock_edge posedge>} : i1
    verif.yield %state : i1
  }
  func.return %bmc : i1
}
