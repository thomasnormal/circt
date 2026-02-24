// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top --max-time=500000000 2>&1 | FileCheck %s
// Regression: standalone sequence assertions should not crash during runtime
// property materialization in circt-sim when no driver process advances time.

module top();
  logic clk;
  logic a;
  logic b;

  sequence seq;
    @(posedge clk) a ##1 b;
  endsequence

  assert property (seq);
endmodule

// CHECK-NOT: unregistered dialect
