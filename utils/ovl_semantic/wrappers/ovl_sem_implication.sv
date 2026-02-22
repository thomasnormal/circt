module ovl_sem_implication(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic antecedent_expr = 1'b1;
`ifdef FAIL
  logic consequent_expr = 1'b0;
`else
  logic consequent_expr = 1'b1;
`endif
  ovl_implication dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .antecedent_expr(antecedent_expr),
      .consequent_expr(consequent_expr),
      .fire());
endmodule
