module top(input logic clk, input logic a, input logic b);
  assert property (@(posedge clk) a |-> b);
endmodule
