// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaDisableIffProceduralMultibit(input logic clk, input logic [1:0] rst,
                                       input logic en, input logic a);
  always @(posedge clk) begin
    if (en) begin
    // CHECK: moore.bool_cast
    // CHECK: moore.to_builtin_bool
    // CHECK: arith.andi
    // CHECK: verif.clocked_assert
    assert property (disable iff (rst) a);

    // CHECK: moore.bool_cast
    // CHECK: moore.to_builtin_bool
    // CHECK: arith.andi
    // CHECK: verif.clocked_assume
    assume property (disable iff (rst) a);
    end
  end
endmodule
