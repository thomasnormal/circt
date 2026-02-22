module ovl_sem_never_unknown(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic qualifier = 1'b1;
`ifdef FAIL
  logic [1:0] test_expr = 2'b0x;
`else
  logic [1:0] test_expr = 2'b01;
`endif
  ovl_never_unknown #(.width(2)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .qualifier(qualifier),
      .test_expr(test_expr),
      .fire());
endmodule
