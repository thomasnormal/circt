// EXPECT_CIRCT_DIAG: driven by always_latch procedure
module t(input logic clk, input logic en, input logic d, output logic q);
  always_ff @(posedge clk) q <= d;
  always_latch if (en) q <= d;
endmodule
