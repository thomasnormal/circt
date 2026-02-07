// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// Test: instance-specific bind through generate hierarchy
//===----------------------------------------------------------------------===//

module dut(input logic clk, input logic [7:0] data);
endmodule

module mon(input logic clk, input logic [7:0] data);
endmodule

module top;
  logic clk;

  generate
    for (genvar i = 0; i < 2; ++i) begin : gen
      logic [7:0] data;
      dut u(.clk(clk), .data(data));
    end
  endgenerate
endmodule

// Bind each generated instance to a distinct monitor.
bind top.gen[0].u mon mon0(.clk(clk), .data(data));
bind top.gen[1].u mon mon1(.clk(clk), .data(data));

// CHECK-LABEL: moore.module private @dut(
// CHECK:         moore.instance "mon0" @mon
// CHECK-LABEL: moore.module private @dut_{{[0-9]+}}(
// CHECK:         moore.instance "mon1" @mon

// CHECK-LABEL: moore.module @top()
// CHECK:         moore.instance "gen_0.u" @dut
// CHECK:         moore.instance "gen_1.u" @dut_{{[0-9]+}}
