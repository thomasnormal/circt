module ovl_sem_unchange(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic [1:0] test_expr = 2'b00;
  logic [1:0] cycles = 2'd0;
  logic start_event;

  assign start_event = (cycles == 2'd1);

  always_ff @(posedge clk) begin
    cycles <= cycles + 2'd1;
`ifdef FAIL
    if (cycles == 2'd1)
      test_expr <= 2'b01;
`endif
  end

  ovl_unchange #(
      .width(2),
      .num_cks(2)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .start_event(start_event),
      .test_expr(test_expr),
      .fire());
endmodule
