module ovl_sem_memory_async(input logic clk);
  logic reset = 1'b1;
  logic enable = 1'b1;
  logic ren;
  logic wen;
  logic [0:0] start_addr = 1'b0;
  logic [0:0] end_addr = 1'b1;
`ifdef FAIL
  logic [0:0] raddr = 1'bx;
`else
  logic [0:0] raddr = 1'b0;
`endif
  logic [0:0] rdata = 1'b0;
  logic [0:0] waddr = 1'b0;
  logic [0:0] wdata = 1'b0;

  assign ren = clk;
  assign wen = clk;

  ovl_memory_async #(
      .data_width(1),
      .addr_width(1),
      .mem_size(2),
      .addr_check(0),
      .init_check(0),
      .one_read_check(0),
      .one_write_check(0),
      .value_check(0)) dut (
      .reset(reset),
      .enable(enable),
      .start_addr(start_addr),
      .end_addr(end_addr),
      .ren(ren),
      .raddr(raddr),
      .rdata(rdata),
      .wen(wen),
      .waddr(waddr),
      .wdata(wdata),
      .fire());
endmodule
