module ovl_sem_never_unknown_async(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
`ifdef FAIL
  logic [1:0] test_expr = 2'b0x;
`else
  logic [1:0] test_expr = 2'b01;
`endif

  ovl_never_unknown_async #(
      .width(2)) dut (
      .reset(reset),
      .enable(enable),
      .test_expr(test_expr),
      .fire());
endmodule
