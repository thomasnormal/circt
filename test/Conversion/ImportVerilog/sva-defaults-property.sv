// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

// CHECK-LABEL: moore.module @test_defaults_property
// CHECK: ltl.or
// CHECK: {sva.disable_iff}
// CHECK-NOT: {sva.disable_iff}
module test_defaults_property(input logic clk, reset, a);
  default clocking @(posedge clk); endclocking
  default disable iff (reset);

  property p;
    a |=> a;
  endproperty

  assert property (p);
endmodule
