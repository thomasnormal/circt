// EXPECT_CIRCT_DIAG: driven by always_comb procedure
module illegal_always_ff_and_always_comb_driver (
  input logic clk,
  input logic a,
  input logic b,
  output logic y
);
  always_ff @(posedge clk)
    y <= a;
  always_comb y = b;
endmodule
