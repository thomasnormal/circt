module ovl_sem_xproduct_bit_coverage(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
`ifdef FAIL
  localparam int cov_check = 1;
  logic test_expr1 = 1'b1;
  logic test_expr2 = 1'b1;
`else
  localparam int cov_check = 0;
  logic test_expr1 = 1'b1;
  logic test_expr2 = 1'b0;
`endif

  ovl_xproduct_bit_coverage #(
      .width1(1),
      .width2(1),
      .test_expr2_enable(1),
      .coverage_check(cov_check)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr1(test_expr1),
      .test_expr2(test_expr2),
      .fire());
endmodule
