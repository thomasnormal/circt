/*
:name: no_property
:description: yosys harness no-property skip
:tags: yosys
*/

module top(input logic clk, input logic in, output logic out);
  always_ff @(posedge clk)
    out <= in;
endmodule
