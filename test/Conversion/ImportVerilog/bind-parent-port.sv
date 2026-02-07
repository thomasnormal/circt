// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang
// XFAIL: *

//===----------------------------------------------------------------------===//
// Test: bind within parent scope referencing parent-only signals
//===----------------------------------------------------------------------===//

module Target(input logic clk);
endmodule

module Monitor(input logic ctrl);
endmodule

module Top(input logic clk, ctrl);
  Target t(.clk(clk));
  // ctrl is not a port of Target; it must resolve in the bind scope (Top).
  bind Target Monitor mon(.ctrl(ctrl));
endmodule

// CHECK-LABEL: moore.module private @Target(
// CHECK:         moore.instance "mon" @Monitor
// CHECK-LABEL: moore.module @Top
// CHECK:         moore.instance "t" @Target
