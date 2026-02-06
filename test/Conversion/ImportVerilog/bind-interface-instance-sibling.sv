// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// Test: bind port connection referencing interface instance in sibling module
//===----------------------------------------------------------------------===//

interface BusIf(input logic clk);
  logic [7:0] data;
endinterface

module Monitor(BusIf bus);
endmodule

module Target;
endmodule

module A(input logic clk);
  BusIf bus(clk);
endmodule

module B;
  Target t();
endmodule

module top;
  logic clk;
  A a(.clk(clk));
  B b();
endmodule

// Bind from the compilation unit to a nested instance; connect to sibling iface.
bind top.b.t Monitor mon(.bus(top.a.bus));

// CHECK-LABEL: moore.module private @Target
// CHECK:         moore.instance "mon" @Monitor
