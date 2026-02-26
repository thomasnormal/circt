// EXPECT_CIRCT_DIAG: cannot mix continuous and procedural assignments
module illegal_output_var_cont_and_proc (
  input logic clk,
  output var logic y
);
  assign y = 1'b0;
  always_ff @(posedge clk)
    y <= 1'b1;
endmodule
