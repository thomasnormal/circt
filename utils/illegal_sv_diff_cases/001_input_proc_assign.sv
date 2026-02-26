// EXPECT_CIRCT_DIAG: cannot assign to input port
module illegal_input_proc_assign (
  input logic clk,
  input logic [7:0] wdata,
  output logic [7:0] q
);
  always_ff @(posedge clk)
    wdata <= q;
endmodule
