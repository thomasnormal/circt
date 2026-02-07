// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// Test: Bind to module definition (affects all instances)
//===----------------------------------------------------------------------===//

module dut1(input clk);
  logic [7:0] data;
endmodule

module monitor_a(input clk, input [7:0] data);
  initial $display("monitor_a");
endmodule

// Bind monitor_a to all instances of dut1
bind dut1 monitor_a mon_a(.clk(clk), .data(data));

// CHECK-LABEL: moore.module private @dut1(in %clk : !moore.l1)
// CHECK:         %data = moore.variable
// CHECK:         moore.instance "mon_a" @monitor_a
// CHECK-LABEL: moore.module private @monitor_a(in %clk : !moore.l1, in %data : !moore.l8)

//===----------------------------------------------------------------------===//
// Test: Instance-specific bind directives
//===----------------------------------------------------------------------===//

module dut2(input clk);
  logic [7:0] data;
endmodule

module monitor_b(input clk, input [7:0] data);
  initial $display("monitor_b");
endmodule

module monitor_c(input clk, input [7:0] data);
  initial $display("monitor_c");
endmodule

// CHECK-LABEL: moore.module private @dut2(in %clk : !moore.l1)
// CHECK:         moore.instance "mon_b" @monitor_b
// CHECK-NOT:     @monitor_c
// CHECK:       }

// CHECK-LABEL: moore.module private @dut2_{{[0-9]+}}(in %clk : !moore.l1)
// CHECK:         moore.instance "mon_c" @monitor_c
// CHECK-NOT:     @monitor_b
// CHECK:       }

//===----------------------------------------------------------------------===//
// Test: Parameterized bind
//===----------------------------------------------------------------------===//

module dut3 #(parameter WIDTH = 8) (input clk, input [WIDTH-1:0] data);
endmodule

module monitor_param #(parameter WIDTH = 8) (input clk, input [WIDTH-1:0] data);
  initial $display("WIDTH=%0d", WIDTH);
endmodule

bind dut3 monitor_param #(.WIDTH(WIDTH)) mon_p(.clk(clk), .data(data));

// CHECK-LABEL: moore.module private @dut3(in %clk : !moore.l1, in %data : !moore.l8)
// CHECK:         moore.instance "mon_p" @monitor_param({{.*}}!moore.l8
// CHECK:       }

// CHECK-LABEL: moore.module private @dut3_{{[0-9]+}}(in %clk : !moore.l1, in %data : !moore.l16)
// CHECK:         moore.instance "mon_p" @monitor_param_{{[0-9]+}}({{.*}}!moore.l16
// CHECK:       }

//===----------------------------------------------------------------------===//
// Test: Multiple binds to same module
//===----------------------------------------------------------------------===//

module dut4(input clk);
  logic [7:0] data;
endmodule

module monitor_d(input clk, input [7:0] data);
endmodule

module monitor_e(input clk, input [7:0] data);
endmodule

bind dut4 monitor_d mon_d(.clk(clk), .data(data));
bind dut4 monitor_e mon_e(.clk(clk), .data(data));

// CHECK-LABEL: moore.module private @dut4(in %clk : !moore.l1)
// CHECK:         moore.instance "mon_d" @monitor_d
// CHECK:         moore.instance "mon_e" @monitor_e
// CHECK:       }

//===----------------------------------------------------------------------===//
// Top module with instances
//===----------------------------------------------------------------------===//

module top;
  logic clk;

  // All instances get monitor_a
  dut1 inst1_a(.clk(clk));
  dut1 inst1_b(.clk(clk));

  // Instance-specific binds
  dut2 inst2_a(.clk(clk));
  dut2 inst2_b(.clk(clk));

  // Parameterized binds
  dut3 #(.WIDTH(8)) inst3_8(.clk(clk), .data(8'h00));
  dut3 #(.WIDTH(16)) inst3_16(.clk(clk), .data(16'h0000));

  // Multiple binds
  dut4 inst4(.clk(clk));
endmodule

// Instance-specific binds for dut2
bind top.inst2_a monitor_b mon_b(.clk(clk), .data(data));
bind top.inst2_b monitor_c mon_c(.clk(clk), .data(data));

// CHECK-LABEL: moore.module @top()
// CHECK:         moore.instance "inst1_a" @dut1
// CHECK:         moore.instance "inst1_b" @dut1
// CHECK:         moore.instance "inst2_a" @dut2(
// CHECK:         moore.instance "inst2_b" @dut2_{{[0-9]+}}(
// CHECK:         moore.instance "inst3_8" @dut3(
// CHECK:         moore.instance "inst3_16" @dut3_{{[0-9]+}}(
// CHECK:         moore.instance "inst4" @dut4
// CHECK:       }
