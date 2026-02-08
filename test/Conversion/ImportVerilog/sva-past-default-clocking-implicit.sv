// RUN: circt-verilog %s --parse-only | FileCheck %s

module test_past_default_clocking_implicit(input logic clk, a);
  default clocking @(posedge clk); endclocking

  property p;
    $past(a);
  endproperty

  assert property (p);
endmodule

// CHECK-LABEL: moore.module @test_past_default_clocking_implicit
// CHECK: moore.past
// CHECK: verif.assert
