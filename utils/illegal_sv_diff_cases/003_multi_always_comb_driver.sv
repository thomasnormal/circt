// EXPECT_CIRCT_DIAG: driven by always_comb procedure
module illegal_multi_always_comb_driver (
  input logic a,
  input logic b,
  output logic y
);
  always_comb y = a;
  always_comb y = b;
endmodule
