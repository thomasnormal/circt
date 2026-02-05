/*
:name: should_fail
:description: minimal should-fail SVA property
:type: simulation elaboration
:tags: 16.10
:should_fail_because: assertion is trivially false
:unsynthesizable: 1
*/

module top(input logic clk);
  property p;
    @(posedge clk) 1'b0;
  endproperty

  assert property (p);
endmodule
