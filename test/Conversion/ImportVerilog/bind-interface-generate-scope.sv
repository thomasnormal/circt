// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// Test: bind inside generate scope referencing interface ports
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
      // Bind in generate scope; connects through interface port in parent.
      bind t Monitor mon(.clk(bus.clk), .data(bus.data));
    end
  endgenerate
endmodule

module top;
  logic clk;
  BusIf bus(clk);
  Wrapper dut(bus);
endmodule

// CHECK-LABEL: moore.module private @Target(
// CHECK:         moore.instance "mon" @Monitor
// CHECK-LABEL: moore.module private @Target_{{[0-9]+}}(
// CHECK:         moore.instance "mon" @Monitor
// CHECK-LABEL: moore.module private @Wrapper
// CHECK:         moore.instance "gen_0.t" @Target
// CHECK:         moore.instance "gen_1.t" @Target_{{[0-9]+}}
