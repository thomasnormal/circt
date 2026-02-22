module ovl_sem_value(input logic clk);
`ifdef FAIL
  logic reset = 1'b1;
`else
  logic reset = 1'b0;
`endif
`ifdef FAIL
  logic enable = 1'b1;
`else
  logic enable = 1'b0;
`endif
  logic disallow = 1'b0;
`ifdef FAIL
  logic [1:0] test_expr = 2'b10;
`else
  logic [1:0] test_expr = 2'b01;
`endif
  logic [3:0] vals = {2'b11, 2'b01};

  ovl_value #(
      .num_values(2),
      .width(2)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr(test_expr),
      .vals(vals),
      .disallow(disallow),
      .fire());
endmodule
