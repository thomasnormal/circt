module ovl_sem_next_state(input logic clk, input logic stim);
  // Keep pass mode reset-gated to avoid end-of-bound over-triggering.
`ifdef FAIL
  logic reset = 1'b1;
`else
  logic reset = 1'b0;
`endif
`ifdef FAIL
  logic enable = 1'b1;
`else
  logic enable = 1'b0;
`endif
  logic test_expr;
  logic curr_state;

  assign test_expr = stim;
  assign curr_state = stim;

`ifdef FAIL
  localparam logic [0:0] allowed_next_state = 1'b0;
  ovl_next_state #(
      .next_count(1),
      .width(1),
      .min_hold(1),
      .max_hold(1),
      .disallow(0)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr(test_expr),
      .curr_state(curr_state),
      .next_state(allowed_next_state),
      .fire());
`else
  localparam logic [1:0] allowed_next_state = 2'b10;
  ovl_next_state #(
      .next_count(2),
      .width(1),
      .min_hold(1),
      .max_hold(1),
      .disallow(0)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr(test_expr),
      .curr_state(curr_state),
      .next_state(allowed_next_state),
      .fire());
`endif
endmodule
