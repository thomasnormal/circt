// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --emit-mlir -b 2 --module=sva_assert_final - | \
// RUN:   FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang
// XFAIL: *

// NOTE: assert final statements are not yet being lowered to BMC assertions

module sva_assert_final(input logic clk, input logic a);
  initial begin
    assert final (a);
  end
endmodule

// CHECK-BMC: verif.assert{{.*}}{bmc.final}
