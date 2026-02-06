// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// Test: instance-specific bind through generate with interface ports
//===----------------------------------------------------------------------===//

interface BusIf(input logic clk);
  logic [7:0] data;
endinterface

module Monitor(input logic clk, input logic [7:0] data);
endmodule

module Target(input logic clk, input logic [7:0] data);
endmodule

module Wrapper(BusIf bus);
  generate
    for (genvar i = 0; i < 2; ++i) begin : gen
      Target t(.clk(bus.clk), .data(bus.data));
    end
  endgenerate
endmodule

module top;
  logic clk;
  BusIf bus(clk);
  Wrapper dut(bus);
endmodule

// Bind from the compilation unit using explicit hierarchical interface paths.
bind top.dut.gen[0].t Monitor mon0(.clk(top.bus.clk), .data(top.bus.data));
bind top.dut.gen[1].t Monitor mon1(.clk(top.bus.clk), .data(top.bus.data));

// CHECK-LABEL: moore.module private @Target(
// CHECK:         moore.instance "mon0" @Monitor
// CHECK-LABEL: moore.module private @Target_{{[0-9]+}}(
// CHECK:         moore.instance "mon1" @Monitor
// CHECK-LABEL: moore.module private @Wrapper
// CHECK:         moore.instance "gen_0.t" @Target
// CHECK:         moore.instance "gen_1.t" @Target_{{[0-9]+}}
