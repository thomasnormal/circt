// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// Test: bind through generate using nested interface instances
//===----------------------------------------------------------------------===//

interface LeafIf #(parameter int AW = 8)(input logic clk);
  logic [AW-1:0] addr;
endinterface

interface TopIf #(parameter int AW = 8)(input logic clk);
  LeafIf #(.AW(AW)) leaf(clk);
endinterface

module Monitor #(parameter int AW = 8)(input logic clk, input logic [AW-1:0] addr);
endmodule

module Target #(parameter int AW = 8)(TopIf bus);
endmodule

module Wrapper #(parameter int AW = 8, int N = 2)(TopIf bus);
  generate
    for (genvar i = 0; i < N; ++i) begin : gen
      Target #(.AW(AW)) t(.bus(bus));
    end
  endgenerate
endmodule

module top;
  logic clk;
  TopIf #(.AW(8)) bus(clk);
  Wrapper #(.AW(8), .N(2)) dut(bus);
endmodule

// Bind using nested interface paths.
bind top.dut.gen[0].t Monitor #(.AW(8)) mon0(.clk(top.bus.leaf.clk), .addr(top.bus.leaf.addr));
bind top.dut.gen[1].t Monitor #(.AW(8)) mon1(.clk(top.bus.leaf.clk), .addr(top.bus.leaf.addr));

// CHECK-LABEL: moore.module private @Target(
// CHECK:         moore.instance "mon0" @Monitor
// CHECK-LABEL: moore.module private @Target_{{[0-9]+}}(
// CHECK:         moore.instance "mon1" @Monitor
// CHECK-LABEL: moore.module private @Wrapper
// CHECK:         moore.instance "gen_0.t" @Target
// CHECK:         moore.instance "gen_1.t" @Target_{{[0-9]+}}
