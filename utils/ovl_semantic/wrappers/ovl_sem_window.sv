module ovl_sem_window(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic start_event = 1'b1;
  logic end_event = 1'b0;
`ifdef FAIL
  logic test_expr = 1'b0;
`else
  logic test_expr = 1'b1;
`endif

  ovl_window dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .start_event(start_event),
      .test_expr(test_expr),
      .end_event(end_event),
      .fire());
endmodule
