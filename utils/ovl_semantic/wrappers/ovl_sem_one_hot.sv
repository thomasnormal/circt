module ovl_sem_one_hot(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
`ifdef FAIL
  logic [3:0] test_expr = 4'b0110;
`else
  logic [3:0] test_expr = 4'b0100;
`endif
  ovl_one_hot #(.width(4)) dut (
      .clock(clk), .reset(reset), .enable(enable), .test_expr(test_expr), .fire());
endmodule
