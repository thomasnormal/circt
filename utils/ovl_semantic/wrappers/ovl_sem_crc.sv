module ovl_sem_crc(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic [0:0] test_expr = 1'b0;
  logic initialize = 1'b0;
  logic valid = 1'b0;
`ifdef FAIL
  logic compare = 1'bx;
`else
  logic compare = 1'b0;
`endif
  logic [4:0] crc = 5'b0;
  logic crc_latch = 1'b0;
  logic [2:0] fire;

  ovl_crc #(
      .width(1),
      .data_width(1),
      .crc_width(5)) dut (
      .clock(clk),
      .reset(reset),
      .enable(enable),
      .test_expr(test_expr),
      .initialize(initialize),
      .valid(valid),
      .compare(compare),
      .crc(crc),
      .crc_latch(crc_latch),
      .fire(fire));
endmodule
