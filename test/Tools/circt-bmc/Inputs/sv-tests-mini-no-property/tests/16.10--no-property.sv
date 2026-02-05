/*
:name: no_property
:description: no property for BMC harness skip handling
:type: simulation elaboration
:tags: 16.10
:unsynthesizable: 1
*/

module top(input logic clk, input logic in, output logic out);
  always_ff @(posedge clk)
    out <= in;
endmodule
