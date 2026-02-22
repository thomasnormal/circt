module ovl_sem_quiescent_state(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic [1:0] check_value = 2'b10;
  logic [1:0] state_expr = 2'b10;
`ifdef FAIL
  logic sample_event = 1'bx;
`else
  logic sample_event = 1'b0;
`endif

  ovl_quiescent_state #(.width(2)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .state_expr(state_expr),
      .check_value(check_value),
      .sample_event(sample_event),
      .fire());
endmodule
