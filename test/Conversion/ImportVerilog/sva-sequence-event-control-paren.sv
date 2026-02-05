// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module sva_sequence_event_control_paren(
  input logic clk,
  input logic req,
  input logic ack
);
  sequence write_on_clk;
    @posedge (clk)
    (req & ack)
  endsequence

  property p_write;
    @(posedge clk) req |-> write_on_clk;
  endproperty

  assert property(p_write);
endmodule
