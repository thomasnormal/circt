// SV equivalent of basic05.vhd — provides the `demo` design module
// that basic05.sv instantiates and asserts properties over.
module demo(input logic clock, input logic ctrl, output logic x);
  logic read = 1'b0, write = 1'b0, ready = 1'b0;
  always_ff @(posedge clock) begin
    read <= ~ctrl;
    write <= ctrl;
    ready <= write;
  end
  assign x = read ^ write ^ ready;
endmodule
