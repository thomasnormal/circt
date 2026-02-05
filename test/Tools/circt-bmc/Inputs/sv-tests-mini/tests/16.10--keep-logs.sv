/*
:name: keep_logs
:description: simple test for KEEP_LOGS flag
:type: simulation elaboration parsing
:tags: 16.10
:unsynthesizable: 1
*/

module top(input logic clk);
  property p;
    @(posedge clk) 1'b1;
  endproperty
  assert property (p);
endmodule
