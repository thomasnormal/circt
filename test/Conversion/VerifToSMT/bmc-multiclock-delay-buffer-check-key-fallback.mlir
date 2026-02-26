// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// Regression: if a check carries an unresolved expression clock key, recover
// the clock position from delay-buffer dependencies.
// CHECK: smt.solver
func.func @bmc_multiclock_delay_buffer_check_key_fallback() -> i1 {
  %res = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk0", "clk1", "sig"],
    bmc_clock_sources = [
      {arg_index = 0 : i32, clock_pos = 0 : i32, invert = false},
      {arg_index = 1 : i32, clock_pos = 1 : i32, invert = false}
    ],
    bmc_clock_keys = ["port:clk0", "port:clk1"]
  } init {
    %f = hw.constant false
    %c0 = seq.to_clock %f
    %c1 = seq.to_clock %f
    verif.yield %c0, %c1 : !seq.clock, !seq.clock
  } loop {
  ^bb0(%c0: !seq.clock, %c1: !seq.clock):
    verif.yield %c0, %c1 : !seq.clock, !seq.clock
  } circuit {
  ^bb0(%c0: !seq.clock, %c1: !seq.clock, %sig: i1):
    %true = ltl.boolean_constant true
    %del = ltl.delay %sig, 1, 0 {bmc.clock = "clk0"} : i1
    %prop = ltl.and %true, %del : !ltl.property, !ltl.sequence
    verif.assert %prop {
      bmc.clock_key = "expr:deadbeef",
      bmc.clock_edge = #ltl<clock_edge posedge>
    } : !ltl.property
    verif.yield %sig : i1
  }
  func.return %res : i1
}

func.func @bmc_multiclock_past_buffer_check_key_fallback() -> i1 {
  %res = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk0", "clk1", "sig"],
    bmc_clock_sources = [
      {arg_index = 0 : i32, clock_pos = 0 : i32, invert = false},
      {arg_index = 1 : i32, clock_pos = 1 : i32, invert = false}
    ],
    bmc_clock_keys = ["port:clk0", "port:clk1"]
  } init {
    %f = hw.constant false
    %c0 = seq.to_clock %f
    %c1 = seq.to_clock %f
    verif.yield %c0, %c1 : !seq.clock, !seq.clock
  } loop {
  ^bb0(%c0: !seq.clock, %c1: !seq.clock):
    verif.yield %c0, %c1 : !seq.clock, !seq.clock
  } circuit {
  ^bb0(%c0: !seq.clock, %c1: !seq.clock, %sig: i1):
    %true = ltl.boolean_constant true
    %past = ltl.past %sig, 1 {bmc.clock = "clk1"} : i1
    %prop = ltl.and %true, %past : !ltl.property, !ltl.sequence
    verif.assert %prop {
      bmc.clock_key = "expr:deadbeef",
      bmc.clock_edge = #ltl<clock_edge posedge>
    } : !ltl.property
    verif.yield %sig : i1
  }
  func.return %res : i1
}
