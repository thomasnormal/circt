module ovl_sem_reg_loaded(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic start_event = 1'b0;
  logic end_event = 1'b0;
  logic [1:0] src_expr = 2'b00;
`ifdef FAIL
  logic [1:0] dest_expr = 2'b01;
`else
  logic [1:0] dest_expr = 2'b00;
`endif
  logic [2:0] cycles = 3'd0;

  always_ff @(posedge clk) begin
    cycles <= cycles + 3'd1;
    start_event <= (cycles == 3'd1);
  end

  ovl_reg_loaded #(
      .width(2),
      .start_count(1),
      .end_count(2)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .start_event(start_event),
      .end_event(end_event),
      .src_expr(src_expr),
      .dest_expr(dest_expr),
      .fire());
endmodule
