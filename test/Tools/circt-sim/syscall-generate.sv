// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// TODO: Generate-for produces data=101 instead of data=0101 — width/padding issue.
// Test generate for/if/case
module top;
  parameter N = 4;
  logic [N-1:0] data;

  genvar i;
  generate
    for (i = 0; i < N; i = i + 1) begin : gen_loop
      assign data[i] = (i % 2 == 0) ? 1'b1 : 1'b0;
    end
  endgenerate

  initial begin
    #1;
    // data should be 0101 (bits 0,2 set)
    // CHECK: data=0101
    $display("data=%b", data);
    $finish;
  end
endmodule
