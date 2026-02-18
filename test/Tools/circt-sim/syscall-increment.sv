// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  logic [7:0] ascending [0:3];     // ascending range: left <= right
  logic [7:0] descending [3:0];    // descending range: left >= right

  initial begin
    // $increment returns 1 if left >= right (descending), -1 if ascending
    // For [0:3]: left=0, right=3, ascending → returns -1
    // CHECK: inc_ascending=-1
    $display("inc_ascending=%0d", $increment(ascending));

    // For [3:0]: left=3, right=0, descending → returns 1
    // CHECK: inc_descending=1
    $display("inc_descending=%0d", $increment(descending));

    $finish;
  end
endmodule
