module ovl_sem_stack(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic push = 1'b0;
`ifdef FAIL
  logic pop = 1'b1;
`else
  logic pop = 1'b0;
`endif
  logic [0:0] push_data = 1'b0;
  logic [0:0] pop_data = 1'b0;
  logic full = 1'b0;
  logic empty = 1'b1;

  ovl_stack #(
      .depth(2),
      .width(1),
      .push_latency(0),
      .pop_latency(0)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .push(push),
      .push_data(push_data),
      .pop(pop),
      .pop_data(pop_data),
      .full(full),
      .empty(empty),
      .fire());
endmodule
