module ovl_sem_always(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
`ifdef FAIL
  logic test_expr = 1'b0;
`else
  logic test_expr = 1'b1;
`endif
  ovl_always dut (
      .clock(clk), .reset(reset), .enable(enable), .test_expr(test_expr), .fire());
endmodule
