// EXPECT_CIRCT_DIAG: driven by always_ff procedure
module illegal_multi_always_ff_driver (
  input logic clk,
  input logic a,
  input logic b,
  output logic y
);
  always_ff @(posedge clk)
    y <= a;
  always_ff @(posedge clk)
    y <= b;
endmodule
