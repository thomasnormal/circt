// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module sva_cover_sequence(input logic clk, a, b);
  // CHECK-LABEL: moore.module @sva_cover_sequence

  // CHECK: verif.cover
  cover sequence (@(posedge clk) a ##1 b);

  // CHECK: verif.clocked_cover
  initial begin
    cover sequence (@(posedge clk) b ##1 a);
  end
endmodule
