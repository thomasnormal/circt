module ovl_sem_zero_one_hot(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
`ifdef FAIL
  logic [2:0] test_expr = 3'b011;  // two asserted bits -> fail
`else
  logic [2:0] test_expr = 3'b001;  // one-hot -> pass
`endif

  ovl_zero_one_hot #(
      .width(3)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr(test_expr),
      .fire());
endmodule
