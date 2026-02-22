module ovl_sem_code_distance(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic [1:0] test_expr1 = 2'b00;
`ifdef FAIL
  logic [1:0] test_expr2 = 2'b0x;
`else
  logic [1:0] test_expr2 = 2'b01;
`endif

  ovl_code_distance #(
      .width(2),
      .min(1),
      .max(1)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr1(test_expr1),
      .test_expr2(test_expr2),
      .fire());
endmodule
