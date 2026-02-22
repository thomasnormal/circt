// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaDisableIffProceduralMultibit(input logic clk, input logic [1:0] rst,
                                       input logic a);
  always @(posedge clk) begin
    // CHECK: moore.bool_cast
    // CHECK: verif.clocked_assert
    assert property (disable iff (rst) a);
  end
endmodule
