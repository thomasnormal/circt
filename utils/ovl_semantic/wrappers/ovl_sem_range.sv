module ovl_sem_range(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
`ifdef FAIL
  logic [3:0] test_expr = 4'd9;
`else
  logic [3:0] test_expr = 4'd4;
`endif
  ovl_range #(.width(4), .min(2), .max(6)) dut (
      .clock(clk), .reset(reset), .enable(enable), .test_expr(test_expr), .fire());
endmodule
