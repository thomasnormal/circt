// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

module test_past_explicit_clock_default_disable(
    input logic clk, fast, reset, en, a);
  default disable iff (reset);

  property p;
    @(posedge clk) $past(a, 2, en, @(posedge fast));
  endproperty

  assert property (p);
endmodule

// CHECK-LABEL: moore.module @test_past_explicit_clock_default_disable
// CHECK-DAG: [[INIT:%[0-9]+]] = moore.constant bX : l1
// CHECK-DAG: [[RESET:%[0-9]+]] = moore.read %reset
// CHECK-DAG: [[EN:%[0-9]+]] = moore.read %en
// CHECK-DAG: moore.wait_event
// CHECK-DAG: moore.conditional [[RESET]] : l1 -> l1
// CHECK-DAG: moore.conditional [[EN]] : l1 -> l1
// CHECK-DAG: moore.yield [[INIT]] : l1
