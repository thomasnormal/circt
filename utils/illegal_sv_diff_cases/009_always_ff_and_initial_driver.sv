// EXPECT_CIRCT_DIAG: driven by always_ff procedure
module illegal_always_ff_and_initial_driver (
  input logic clk,
  input logic a,
  output logic y
);
  always_ff @(posedge clk)
    y <= a;
  initial y = 1'b0;
endmodule
