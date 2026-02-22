module ovl_sem_increment(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic [3:0] test_expr = 4'd0;
  logic [1:0] cycles = 2'd0;

  always_ff @(posedge clk) begin
    cycles <= cycles + 2'd1;
    if (cycles == 2'd1) begin
`ifdef FAIL
      test_expr <= 4'd2;
`else
      test_expr <= 4'd1;
`endif
    end
  end

  ovl_increment #(
      .width(4),
      .value(1)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr(test_expr),
      .fire());
endmodule
