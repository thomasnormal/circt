// EXPECT_CIRCT_DIAG: driven by always_ff procedure
module t(input logic clk, input logic a, input logic b, output logic q);
  if (1) begin : g1
    always_ff @(posedge clk) q <= a;
  end
  if (1) begin : g2
    always_ff @(posedge clk) q <= b;
  end
endmodule
