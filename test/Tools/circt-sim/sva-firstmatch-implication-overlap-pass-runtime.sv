// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=200000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS_FIRSTMATCH_OVERLAP
// CHECK-NOT: SVA assertion failed at time

// Overlapping antecedent triggers should not induce false failures when
// first_match can satisfy each bounded obligation.

module top;
  reg clk;
  reg a;
  reg b;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b1;
    b = 1'b1;

    // Keep antecedent active for several sampled cycles.
    repeat (4) @(posedge clk);
    a = 1'b0;

    // Allow previously-triggered bounded obligations to close cleanly.
    repeat (4) @(posedge clk);

    $display("SVA_PASS_FIRSTMATCH_OVERLAP");
    $finish;
  end

  assert property (@(posedge clk) a |-> first_match(##[1:2] b));
endmodule
