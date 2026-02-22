module ovl_sem_multiport_fifo(input logic clk);
  logic reset = 1'b0;
  logic enable = 1'b1;
`ifdef FAIL
  logic [1:0] enq = 2'b0x;
`else
  logic [1:0] enq = 2'b00;
`endif
  logic [1:0] deq = 2'b00;
  logic [1:0] enq_data = 2'b00;
  logic [1:0] deq_data = 2'b00;
  logic full = 1'b0;
  logic empty = 1'b1;
  logic [0:0] preload = 1'b0;

  always_ff @(posedge clk)
    reset <= 1'b1;

  ovl_multiport_fifo #(
      .width(1),
      .depth(2),
      .enq_count(2),
      .deq_count(2),
      .value_check(0),
      .full_check(0),
      .empty_check(0),
      .preload_count(0)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .enq(enq),
      .deq(deq),
      .enq_data(enq_data),
      .deq_data(deq_data),
      .full(full),
      .empty(empty),
      .preload(preload),
      .fire());
endmodule
