// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// Test: bind through parameterized generate with interface ports
//===----------------------------------------------------------------------===//

interface BusIf(input logic clk);
  logic [7:0] data;
endinterface

module Monitor #(parameter int ID = 0)(input logic clk, input logic [7:0] data);
endmodule

module Target #(parameter int ID = 0)(input logic clk, input logic [7:0] data);
endmodule

module Wrapper #(parameter int N = 2)(BusIf bus);
  generate
    for (genvar i = 0; i < N; ++i) begin : gen
      Target #(.ID(i)) t(.clk(bus.clk), .data(bus.data));
    end
  endgenerate
endmodule

module top;
  logic clk;
  BusIf bus(clk);
  Wrapper #(.N(2)) dut(bus);
endmodule

// Bind from the compilation unit using explicit hierarchical interface paths.
bind top.dut.gen[0].t Monitor #(.ID(0)) mon0(.clk(top.bus.clk), .data(top.bus.data));
bind top.dut.gen[1].t Monitor #(.ID(1)) mon1(.clk(top.bus.clk), .data(top.bus.data));

// CHECK-LABEL: moore.module private @Target(
// CHECK:         moore.instance "mon0" @Monitor
// CHECK-LABEL: moore.module private @Target_{{[0-9]+}}(
// CHECK:         moore.instance "mon1" @Monitor_{{[0-9]+}}
// CHECK-LABEL: moore.module private @Wrapper
// CHECK:         moore.instance "gen_0.t" @Target
// CHECK:         moore.instance "gen_1.t" @Target_{{[0-9]+}}
