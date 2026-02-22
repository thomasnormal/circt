// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVABoundedEventuallyProperty(input logic clk, a, b, c, d);
  property p;
    @(posedge clk) a |-> b;
  endproperty

  // Bounded eventually on a property operand should lower without invalid
  // sequence-only ops on `!ltl.property`.
  // CHECK-LABEL: moore.module @SVABoundedEventuallyProperty
  // CHECK: ltl.implication
  // CHECK: ltl.or
  // CHECK: verif.assert
  assert property (eventually [1:2] p);

  // Strong bounded eventually should follow the same bounded lowering shape.
  // CHECK: ltl.implication
  // CHECK: ltl.or
  // CHECK: verif.assert
  assert property (s_eventually [2:3] p);

  // Keep a direct property use nearby as a guard.
  // CHECK: verif.assert
  assert property (@(posedge clk) c |-> d);
endmodule
