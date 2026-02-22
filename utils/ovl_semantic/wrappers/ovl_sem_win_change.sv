module ovl_sem_win_change(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
`ifdef FAIL
  logic start_event = 1'b1;
  logic end_event = 1'b1;
`else
  logic start_event = 1'b0;
  logic end_event = 1'b0;
`endif
  logic test_expr = 1'b0;

  ovl_win_change #(.width(1)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .start_event(start_event),
      .test_expr(test_expr),
      .end_event(end_event),
      .fire());
endmodule
