// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang
// Bind with interface port access - using bind scope's interface ports.

interface BusIf(input logic clk);
  logic data;
endinterface

module Monitor(input logic clk, input logic data);
endmodule

module Target(input logic clk, input logic data);
endmodule

module Wrapper(BusIf bus);
  Target t0(.clk(bus.clk), .data(bus.data));

  // Bind in wrapper scope; connects through interface port.
  bind Target Monitor mon(.clk(bus.clk), .data(bus.data));
endmodule

module top;
  logic clk;
  BusIf bus(clk);
  Wrapper dut(bus);
endmodule

// CHECK-LABEL: moore.module private @Monitor
// CHECK-LABEL: moore.module private @Target
// CHECK:         moore.instance "mon" @Monitor
// CHECK-LABEL: moore.module private @Wrapper
// CHECK:         moore.instance "t0" @Target
// CHECK-LABEL: moore.module @top
