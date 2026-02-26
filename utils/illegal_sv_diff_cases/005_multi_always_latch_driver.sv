// EXPECT_CIRCT_DIAG: driven by always_latch procedure
module illegal_multi_always_latch_driver (
  input logic en1,
  input logic en2,
  input logic a,
  input logic b,
  output logic y
);
  always_latch if (en1) y <= a;
  always_latch if (en2) y <= b;
endmodule
