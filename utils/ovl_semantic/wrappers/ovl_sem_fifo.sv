module ovl_sem_fifo(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic enq = 1'b0;
  logic deq = 1'b0;
`ifdef FAIL
  logic full = 1'bx;
`else
  logic full = 1'b0;
`endif
  logic empty = 1'b1;
  logic [0:0] enq_data = 1'b0;
  logic [0:0] deq_data = 1'b0;
  logic [0:0] preload = 1'b0;

  ovl_fifo #(
      .depth(2),
      .width(1),
      .value_check(0),
      .preload_count(0)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .enq(enq),
      .enq_data(enq_data),
      .deq(deq),
      .deq_data(deq_data),
      .full(full),
      .empty(empty),
      .preload(preload),
      .fire());
endmodule
