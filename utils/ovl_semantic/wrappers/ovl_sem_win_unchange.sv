module ovl_sem_win_unchange(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
`ifdef FAIL
  logic start_event = 1'b1;
`else
  logic start_event = 1'b0;
`endif
  logic end_event = 1'b0;
  logic test_expr = 1'b0;
  logic [2:0] cycles = 3'd0;

  always_ff @(posedge clk) begin
    cycles <= cycles + 3'd1;
`ifdef FAIL
    if (cycles == 3'd1)
      test_expr <= 1'b1;
`endif
  end

  ovl_win_unchange #(.width(1)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .start_event(start_event),
      .test_expr(test_expr),
      .end_event(end_event),
      .fire());
endmodule
