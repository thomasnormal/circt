// EXPECT_CIRCT_DIAG: cannot assign to a net within a procedural context
module illegal_wire_proc_assign (
  input logic a,
  output wire y
);
  always_comb
    y = a;
endmodule
