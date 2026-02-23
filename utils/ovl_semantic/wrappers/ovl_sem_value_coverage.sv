module ovl_sem_value_coverage(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
`ifdef FAIL
  logic test_expr = 1'bx;
`else
  logic test_expr = 1'b0;
`endif
  logic is_not = 1'b0;

  ovl_value_coverage #(
      .width(1),
      .is_not_width(1),
      .is_not_count(0),
      .value_coverage(1)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr(test_expr),
      .is_not(is_not),
      .fire());
endmodule
