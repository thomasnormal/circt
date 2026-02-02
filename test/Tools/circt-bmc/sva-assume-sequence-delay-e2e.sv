// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 5 --module=sva_assume_sequence_delay - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

// End-to-end test for sequence assumptions with delays (##1).
// This tests that assume constraints apply from cycle 0, matching Yosys
// behavior with `-early -assume`.
//
// From yosys/tests/sva/sva_value_change_changed.sv:
//   assert property ($changed(b));
//   assume property (b !== x ##1 $changed(b));
//
// The assume has ##1 which creates a 2-cycle sequence. The assumption should
// constrain b to change every cycle starting from cycle 0. Without the
// skipWarmup fix, the NFA wouldn't constrain cycle 0 and the assert would fail.

// CHECK: BMC_RESULT=UNSAT

module sva_assume_sequence_delay(input logic clk, input logic b);
  wire x = 'x;

  // This should PASS: the assumption constrains b to change every cycle
  assert property (@(posedge clk) $changed(b));

  // This assumption has ##1 delay - should constrain from cycle 0
  assume property (@(posedge clk) b !== x ##1 $changed(b));

endmodule
