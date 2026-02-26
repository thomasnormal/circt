// EXPECT_CIRCT_DIAG: cannot mix continuous and procedural assignments
module illegal_proc_and_cont_driver (
  input logic clk,
  input logic a,
  output logic y
);
  assign y = a;
  always_ff @(posedge clk)
    y <= ~a;
endmodule
