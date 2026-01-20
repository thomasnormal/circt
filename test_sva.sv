module test_sva;
  logic clk, rst, a, b;

  property p_ab;
    @(posedge clk) disable iff (rst)
    a |-> ##1 b;
  endproperty

  assert property (p_ab);
endmodule
