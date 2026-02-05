// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --emit-mlir -b 2 --module=sva_expect - | \
// RUN:   FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang
// XFAIL: *

// NOTE: expect statements are not yet being lowered to BMC

module sva_expect(input logic clk, input logic a, input logic b);
  initial begin
    expect (@(posedge clk) a ##1 b);
  end
endmodule

// CHECK-BMC: verif.bmc
