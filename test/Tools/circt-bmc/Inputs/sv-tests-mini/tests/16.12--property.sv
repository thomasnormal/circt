/*
:name: property_test
:description: bare property test
:tags: 16.12
:unsynthesizable: 1
*/

module top(input logic clk, input logic a);
  assert property (@(posedge clk) (a == 1'b1));
endmodule
