// SV equivalent of basic04.vhd — provides the `top` design module
// that basic04.sv binds SVA properties into.
module top(input logic clock, input logic ctrl, output logic x);
  logic read = 1'b0, write = 1'b0, ready = 1'b0;
  always_ff @(posedge clock) begin
    read <= ~ctrl;
    write <= ctrl;
    ready <= write;
  end
  assign x = read ^ write ^ ready;
endmodule
