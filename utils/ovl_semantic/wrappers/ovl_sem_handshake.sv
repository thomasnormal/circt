module ovl_sem_handshake(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic req = 1'b0;
  logic ack = 1'b0;
  logic [2:0] cycles = 3'd0;

  always_ff @(posedge clk) begin
    cycles <= cycles + 3'd1;
`ifdef FAIL
    case (cycles)
      3'd1: ack <= 1'b1;
      3'd2: ack <= 1'b0;
      default: ack <= 1'b0;
    endcase
`else
    ack <= 1'b0;
`endif
  end

  ovl_handshake #(
      .min_ack_cycle(1)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .req(req),
      .ack(ack),
      .fire());
endmodule
