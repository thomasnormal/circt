module ovl_sem_fifo_index(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic [0:0] push = 1'b0;
`ifdef FAIL
  logic [0:0] pop = 1'b1;
`else
  logic [0:0] pop = 1'b0;
`endif

  ovl_fifo_index #(
      .depth(2),
      .push_width(1),
      .pop_width(1),
      .simultaneous_push_pop(1)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .push(push),
      .pop(pop),
      .fire());
endmodule
