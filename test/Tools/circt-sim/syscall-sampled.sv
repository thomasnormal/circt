// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $sampled — returns the sampled value of its argument in procedural context.
// In non-assertion context, $sampled simply returns the current value.
module top;
  reg [7:0] val = 0;

  initial begin
    val = 8'hFF;
    // CHECK: sampled=255
    $display("sampled=%0d", $sampled(val));

    val = 8'h42;
    // CHECK: sampled2=66
    $display("sampled2=%0d", $sampled(val));

    // CHECK: done
    $display("done");
    $finish;
  end
endmodule
