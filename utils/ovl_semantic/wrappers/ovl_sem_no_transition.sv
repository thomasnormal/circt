module ovl_sem_no_transition(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic [1:0] test_expr = 2'b00;
  logic [2:0] cycles = 3'd0;

  localparam logic [1:0] start_state = 2'b01;
  localparam logic [1:0] next_state = 2'b10;

  always_ff @(posedge clk) begin
    cycles <= cycles + 3'd1;
    if (cycles == 3'd1)
      test_expr <= start_state;
    else if (cycles == 3'd2) begin
`ifdef FAIL
      test_expr <= next_state;
`else
      test_expr <= 2'b11;
`endif
    end
  end

  ovl_no_transition #(.width(2)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr(test_expr),
      .start_state(start_state),
      .next_state(next_state),
      .fire());
endmodule
