module ovl_sem_odd_parity(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
`ifdef FAIL
  logic [3:0] test_expr = 4'b0011;
`else
  logic [3:0] test_expr = 4'b0001;
`endif

  ovl_odd_parity #(.width(4)) dut (
      .clock(clk), .reset(reset), .enable(enable), .test_expr(test_expr), .fire());
endmodule
