// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// Regression: imported checks may carry canonicalized clock metadata (clk_i)
// while the BMC clock input name is widened to clk_io_i. Accept a unique
// name/key alias and preserve conversion.
// CHECK: smt.solver
func.func @bmc_clock_name_io_suffix_alias() -> i1 {
  %res = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk_io_i", "sig"],
    bmc_clock_sources = [
      {arg_index = 0 : i32, clock_pos = 0 : i32, invert = false}
    ],
    bmc_clock_keys = ["port:clk_io_i"]
  } init {
    %f = hw.constant false
    %c0 = seq.to_clock %f
    verif.yield %c0 : !seq.clock
  } loop {
  ^bb0(%c0: !seq.clock):
    verif.yield %c0 : !seq.clock
  } circuit {
  ^bb0(%c0: !seq.clock, %sig: i1):
    %t = hw.constant true
    verif.assert %t {
      bmc.clock = "clk_i",
      bmc.clock_key = "port:clk_i",
      bmc.clock_edge = #ltl<clock_edge posedge>
    } : i1
    verif.yield %sig : i1
  }
  func.return %res : i1
}
