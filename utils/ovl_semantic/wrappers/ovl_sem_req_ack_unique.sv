module ovl_sem_req_ack_unique(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic req = 1'b0;
`ifdef FAIL
  logic ack = 1'b1;
`else
  logic ack = 1'b0;
`endif

  ovl_req_ack_unique #(
      .min_cks(1),
      .max_cks(4),
      .method(0)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .req(req),
      .ack(ack),
      .fire());
endmodule
