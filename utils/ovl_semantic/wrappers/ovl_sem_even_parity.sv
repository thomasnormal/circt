module ovl_sem_even_parity(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
`ifdef FAIL
  logic [2:0] test_expr = 3'b001;  // odd parity -> fail
`else
  logic [2:0] test_expr = 3'b011;  // even parity -> pass
`endif

  ovl_even_parity #(
      .width(3)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr(test_expr),
      .fire());
endmodule
