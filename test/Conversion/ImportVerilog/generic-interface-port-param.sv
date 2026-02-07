// RUN: circt-verilog --ir-moore %s --no-uvm-auto-include | FileCheck %s
// REQUIRES: slang

// Test generic interface port with parameterized interface.
// This is the pattern used by AXI-VIP and similar verification IPs.

interface param_if #(parameter WIDTH = 8);
  logic [WIDTH-1:0] data;
  logic valid;
endinterface

// Module with a generic interface port connected to parameterized interface
module consumer(interface bus);
  logic out;
  assign out = bus.valid;
endmodule

// CHECK-LABEL: moore.module @top
module top;
  param_if #(.WIDTH(16)) wide_bus();

  // CHECK: moore.instance "c" @consumer
  consumer c(.bus(wide_bus));
endmodule
