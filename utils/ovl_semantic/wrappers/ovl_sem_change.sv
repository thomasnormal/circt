module ovl_sem_change(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
`ifdef FAIL
  logic start_event = 1'b1;
`else
  logic start_event = 1'b0;
`endif
  logic [1:0] test_expr = 2'b00;

  ovl_change #(
      .width(2),
      .num_cks(2)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .start_event(start_event),
      .test_expr(test_expr),
      .fire());
endmodule
