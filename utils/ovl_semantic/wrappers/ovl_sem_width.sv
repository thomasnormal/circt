module ovl_sem_width(input logic clk);
  logic reset = 1'b1;
`ifdef FAIL
  logic enable = 1'b1;
`else
  logic enable = 1'b0;
`endif
  logic [3:0] cycles = 4'd0;
  logic test_expr;

  always_ff @(posedge clk)
    cycles <= cycles + 4'd1;

`ifdef FAIL
  assign test_expr = (cycles >= 4'd1 && cycles <= 4'd3);
`else
  assign test_expr = 1'b0;
`endif

  ovl_width #(
      .min_cks(2),
      .max_cks(2)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr(test_expr),
      .fire());
endmodule
