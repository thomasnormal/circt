// EXPECT_CIRCT_DIAG: driven by always_ff procedure
module t(input logic clk, input logic a, output logic q);
  always_ff @(posedge clk) q <= a;
  always @* q = a;
endmodule
