/*
:name: property_disj_test
:description: bare property disjunction test
:tags: 16.12
:unsynthesizable: 1
*/

module top(input logic clk, input logic a, input logic b);
  assert property (@(posedge clk) (a || b));
endmodule
