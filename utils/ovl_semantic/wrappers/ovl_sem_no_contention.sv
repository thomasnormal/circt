module ovl_sem_no_contention(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic [1:0] test_expr = 2'b10;
`ifdef FAIL
  logic [1:0] driver_enables = 2'b11;
`else
  logic [1:0] driver_enables = 2'b01;
`endif

  ovl_no_contention #(
      .min_quiet(1),
      .max_quiet(1),
      .num_drivers(2),
      .width(2)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr(test_expr),
      .driver_enables(driver_enables),
      .fire());
endmodule
