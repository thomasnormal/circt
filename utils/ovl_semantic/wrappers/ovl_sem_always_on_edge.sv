`include "std_ovl_defines.h"

module ovl_sem_always_on_edge(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic [2:0] cycles = 3'd0;
  logic sampling_event;
  logic test_expr;

  always_ff @(posedge clk)
    cycles <= cycles + 3'd1;

  assign sampling_event = (cycles == 3'd1);
`ifdef FAIL
  assign test_expr = (cycles == 3'd1) ? 1'b0 : 1'b1;
`else
  assign test_expr = 1'b1;
`endif

  ovl_always_on_edge #(
      .edge_type(`OVL_POSEDGE)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .sampling_event(sampling_event),
      .test_expr(test_expr),
      .fire());
endmodule
