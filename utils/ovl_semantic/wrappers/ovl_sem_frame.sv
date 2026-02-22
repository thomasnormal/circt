`include "std_ovl_defines.h"

module ovl_sem_frame(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic start_event = 1'b1;
`ifdef FAIL
  logic test_expr = 1'b0;
`else
  logic test_expr = 1'b1;
`endif

  ovl_frame #(
      .min_cks(0),
      .max_cks(0),
      .action_on_new_start(`OVL_IGNORE_NEW_START)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .start_event(start_event),
      .test_expr(test_expr),
      .fire());
endmodule
