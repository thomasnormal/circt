/*
:name: macro_test
:description: macro-gated module for verilog-args test
:tags: 16.10
:unsynthesizable: 1
*/

`ifdef ENABLE
module top(input logic clk);
  property p;
    @(posedge clk) 1'b1;
  endproperty
  assert property (p);
endmodule
`endif
