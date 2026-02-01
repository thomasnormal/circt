module top(
  input logic clk,
  input logic reset,
  input logic antecedent,
  output logic consequent
);
  always @(posedge clk)
    consequent <= reset ? 1'b0 : antecedent;

  assert property (@(posedge clk) disable iff (reset) antecedent |-> consequent);
endmodule
