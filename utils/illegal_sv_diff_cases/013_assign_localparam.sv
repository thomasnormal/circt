// EXPECT_CIRCT_DIAG: is not modifiable
module illegal_assign_localparam;
  localparam int P = 0;
  initial
    P = 1;
endmodule
