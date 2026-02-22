module ovl_sem_no_underflow(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic [3:0] test_expr = 4'd3;
  logic [2:0] cycles = 3'd0;

  always_ff @(posedge clk) begin
    cycles <= cycles + 3'd1;
    if (cycles == 3'd1)
      test_expr <= 4'd1;
    else if (cycles == 3'd2) begin
`ifdef FAIL
      test_expr <= 4'd6;
`else
      test_expr <= 4'd2;
`endif
    end
  end

  ovl_no_underflow #(
      .width(4),
      .min(1),
      .max(5)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr(test_expr),
      .fire());
endmodule
