// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

module test_past_default_disable_reset(input logic clk, reset, a);
  default clocking @(posedge clk); endclocking
  default disable iff (reset);

  property p;
    $past(a, 2);
  endproperty

  assert property (p);
endmodule

// CHECK-LABEL: moore.module @test_past_default_disable_reset
// CHECK-DAG: [[INIT:%[0-9]+]] = moore.constant bX : l1
// CHECK: moore.procedure always
// CHECK: moore.wait_event
// CHECK-DAG: [[RESET:%[0-9]+]] = moore.read %reset
// CHECK: [[ENABLE:%[0-9]+]] = moore.not [[RESET]] : l1
// CHECK: moore.conditional [[ENABLE]] : l1 -> l1
// CHECK: moore.yield [[INIT]] : l1
