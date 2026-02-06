// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

module test_past_default_disable_enable(input logic clk, reset, en, a);
  default clocking @(posedge clk); endclocking
  default disable iff (reset);

  property p;
    $past(a, 2, en);
  endproperty

  assert property (p);
endmodule

// CHECK-LABEL: moore.module @test_past_default_disable_enable
// CHECK-DAG: [[INIT:%[0-9]+]] = moore.constant bX : l1
// CHECK-DAG: [[RESET:%[0-9]+]] = moore.read %reset
// CHECK-DAG: [[EN:%[0-9]+]] = moore.read %en
// CHECK-DAG: moore.conditional [[RESET]] : l1 -> l1
// CHECK-DAG: moore.conditional [[EN]] : l1 -> l1
// CHECK-DAG: moore.yield [[INIT]] : l1
