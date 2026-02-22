module ovl_sem_bits(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
`ifdef FAIL
  logic [1:0] test_expr = 2'b11;
`else
  logic [1:0] test_expr = 2'b01;
`endif

  ovl_bits #(
      .width(2),
      .min(1),
      .max(1),
      .asserted(1)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr(test_expr),
      .fire());
endmodule
