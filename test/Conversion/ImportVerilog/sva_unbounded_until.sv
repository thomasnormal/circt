// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module RangeUntil(input logic clk, a, b, c, d);
  // Unbounded delay sequence with until in the consequent.
  property p_range_until;
    @(posedge clk) a ##[*] b |=> c until d;
  endproperty
  assert property (p_range_until);
endmodule

// CHECK-LABEL: moore.module @RangeUntil
// CHECK: ltl.delay {{%[a-z0-9]+}}, 0 : i1
// CHECK: ltl.concat
// CHECK: ltl.until
// CHECK: ltl.implication
// CHECK: verif.{{(clocked_)?}}assert
