module ovl_sem_cycle_sequence(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
`ifdef FAIL
  logic [1:0] event_sequence = 2'bx0;
`else
  logic [1:0] event_sequence = 2'b00;
`endif
  logic [2:0] cycles = 3'd0;

  always_ff @(posedge clk) begin
    cycles <= cycles + 3'd1;
`ifndef FAIL
    event_sequence <= 2'b00;
`endif
  end

  ovl_cycle_sequence #(
      .num_cks(2)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .event_sequence(event_sequence),
      .fire());
endmodule
