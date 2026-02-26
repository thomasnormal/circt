// EXPECT_CIRCT_DIAG: always_ff procedure must have one and only one event control
module t(input logic a, output logic q);
  always_ff q <= a;
endmodule
