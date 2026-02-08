// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test clocking block event reference syntax: @(cb)
// This syntax waits for the clocking block's clock event.

module test_clocking_event;
  logic clk, data;

  // Clocking block with posedge clock event
  clocking cb @(posedge clk);
    input data;
  endclocking

  // Test @(cb) syntax - should wait for the clocking block's clock event
  // CHECK-LABEL: moore.procedure always
  // CHECK: moore.wait_event {
  // CHECK-NEXT: %[[CLK:.+]] = moore.read %clk
  // CHECK-NEXT: moore.detect_event posedge %[[CLK]]
  // CHECK-NEXT: }
  always @(cb) begin
    $display("Data: %0d", cb.data);
  end
endmodule

// Test with negedge clocking block
module test_clocking_event_negedge;
  logic clk, data;

  clocking cb @(negedge clk);
    input data;
  endclocking

  // CHECK-LABEL: moore.procedure always
  // CHECK: moore.wait_event {
  // CHECK-NEXT: %[[CLK:.+]] = moore.read %clk
  // CHECK-NEXT: moore.detect_event negedge %[[CLK]]
  // CHECK-NEXT: }
  always @(cb) begin
    $display("Data: %0d", cb.data);
  end
endmodule

// Concurrent assertions inside always @(cb) should canonicalize the clocking
// block event to its underlying signal event when hoisted.
module test_clocking_event_assert;
  logic clk, a;

  clocking cb @(posedge clk);
    input a;
  endclocking

  always @(cb) begin
    assert property ($rose(a));
  end
endmodule

// CHECK-LABEL: moore.module @test_clocking_event_assert
// CHECK: verif.clocked_assert
// CHECK-SAME: posedge
