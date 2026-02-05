/*
:name: multiclock_min
:description: minimal multi-clock SVA property
:type: simulation elaboration
:tags: 16.13
:unsynthesizable: 1
*/

module top(input logic clk0, input logic clk1);
  property p0;
    @(posedge clk0) 1'b1;
  endproperty
  property p1;
    @(posedge clk1) 1'b1;
  endproperty

  assert property (p0);
  assert property (p1);
endmodule
