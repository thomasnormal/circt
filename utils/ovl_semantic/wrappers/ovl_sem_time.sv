module ovl_sem_time(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic [2:0] cycles = 3'd0;
  logic start_event;
  logic test_expr;

  always_ff @(posedge clk)
    cycles <= cycles + 3'd1;

  assign start_event = (cycles == 3'd1);
`ifdef FAIL
  assign test_expr = 1'b0;
`else
  assign test_expr = 1'b1;
`endif

  ovl_time #(
      .num_cks(2)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .start_event(start_event),
      .test_expr(test_expr),
      .fire());
endmodule
