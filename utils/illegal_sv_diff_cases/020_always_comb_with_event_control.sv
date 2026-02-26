// EXPECT_CIRCT_DIAG: statements that pass time are not allowed
module t(input logic clk, input logic a, output logic q);
  always_comb @(posedge clk) q = a;
endmodule
