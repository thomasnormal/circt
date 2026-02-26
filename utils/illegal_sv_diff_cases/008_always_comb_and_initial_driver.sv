// EXPECT_CIRCT_DIAG: driven by always_comb procedure
module illegal_always_comb_and_initial_driver (
  input logic a,
  output logic y
);
  always_comb y = a;
  initial y = 1'b0;
endmodule
