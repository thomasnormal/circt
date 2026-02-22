module ovl_sem_arbiter(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic [1:0] reqs = 2'b01;
`ifdef FAIL
  logic [1:0] gnts = 2'b11;
`else
  logic [1:0] gnts = 2'b01;
`endif
  logic [1:0] priorities = 2'b01;

  ovl_arbiter #(
      .width(2),
      .priority_width(1),
      .min_cks(0),
      .max_cks(0),
      .priority_check(0),
      .arbitration_rule(0),
      .one_cycle_gnt_check(0)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .reqs(reqs),
      .priorities(priorities),
      .gnts(gnts),
      .fire());
endmodule
