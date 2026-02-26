// EXPECT_CIRCT_DIAG: multiple continuous assignments to variable
module illegal_output_var_two_cont_assigns (
  input logic a,
  input logic b,
  output var logic y
);
  assign y = a;
  assign y = b;
endmodule
