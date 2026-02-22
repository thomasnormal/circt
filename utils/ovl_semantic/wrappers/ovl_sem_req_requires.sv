module ovl_sem_req_requires(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic req_trigger = 1'b1;
`ifdef FAIL
  logic req_follower = 1'b0;
  logic resp_leader = 1'b0;
  logic resp_trigger = 1'b0;
`else
  logic req_follower = 1'b1;
  logic resp_leader = 1'b1;
  logic resp_trigger = 1'b1;
`endif

  ovl_req_requires #(
      .min_cks(1),
      .max_cks(6)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .req_trigger(req_trigger),
      .req_follower(req_follower),
      .resp_leader(resp_leader),
      .resp_trigger(resp_trigger),
      .fire());
endmodule
