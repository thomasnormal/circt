// RUN: circt-verilog --ir-moore --no-uvm-auto-include -DADDR_WIDTH=32 -DDATA_WIDTH=32 %s
// RUN: circt-verilog --ir-moore --no-uvm-auto-include -DADDR_WIDTH=64 -DDATA_WIDTH=64 %s
// REQUIRES: slang

`ifdef ADDR_WIDTH == 32 && DATA_WIDTH == 32
module pp_ifdef_expr;
  logic a;
endmodule
`elsif ADDR_WIDTH == 64 && DATA_WIDTH == 64
module pp_ifdef_expr;
  logic b;
endmodule
`endif
