module ovl_sem_hold_value(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic [1:0] value = 2'b10;
  logic [1:0] test_expr = 2'b00;
  logic [2:0] cycles = 3'd0;

  always_ff @(posedge clk) begin
    cycles <= cycles + 3'd1;
    if (cycles == 3'd1)
      test_expr <= value;
    else if (cycles == 3'd2) begin
`ifdef FAIL
      test_expr <= value;
`else
      test_expr <= 2'b11;
`endif
    end
  end

  ovl_hold_value #(
      .min(0),
      .max(0),
      .width(2)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr(test_expr),
      .value(value),
      .fire());
endmodule
