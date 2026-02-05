module top(input logic clk, input logic a);
  property p;
    @(posedge clk) a;
  endproperty
  assert property (p);
endmodule
