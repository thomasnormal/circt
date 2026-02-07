// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc --emit-mlir -b 2 --module=sva_expect - | \
// RUN:   FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang

module sva_expect(input logic clk, input logic a, input logic b);
  initial begin
    expect (@(posedge clk) a ##1 b);
  end
endmodule

// CHECK-BMC: func.func @sva_expect()
// CHECK-BMC: scf.for
// CHECK-BMC: smt.assert
