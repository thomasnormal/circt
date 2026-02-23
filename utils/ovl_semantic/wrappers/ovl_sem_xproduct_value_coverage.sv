module ovl_sem_xproduct_value_coverage(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic val1 = 1'b1;
  logic val2 = 1'b1;
`ifdef FAIL
  localparam int cov_check = 1;
  logic test_expr1 = 1'b1;
  logic test_expr2 = 1'b1;
`else
  localparam int cov_check = 0;
  logic test_expr1 = 1'b1;
  logic test_expr2 = 1'b0;
`endif

  ovl_xproduct_value_coverage #(
      .width1(1),
      .width2(1),
      .val1_width(1),
      .val2_width(1),
      .val1_count(1),
      .val2_count(1),
      .coverage_check(cov_check)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr1(test_expr1),
      .test_expr2(test_expr2),
      .val1(val1),
      .val2(val2),
      .fire());
endmodule
