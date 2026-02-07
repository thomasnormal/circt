// RUN: circt-verilog %s --ir-moore 2>&1 | FileCheck %s

// Test that different parameterizations of the same interface produce separate
// MLIR interface declarations, while identical parameterizations are deduplicated.

interface param_if #(parameter WIDTH = 8);
  logic [WIDTH-1:0] data;
  logic valid;
endinterface

module top;
  param_if #(.WIDTH(16)) narrow_bus();
  param_if #(.WIDTH(32)) wide_bus();
  param_if #(.WIDTH(16)) another_narrow_bus();
endmodule

// CHECK: moore.interface @param_if {
// CHECK:   moore.interface.signal @data : !moore.l16
// CHECK:   moore.interface.signal @valid : !moore.l1
// CHECK: }

// CHECK: moore.interface @param_if_1 {
// CHECK:   moore.interface.signal @data : !moore.l32
// CHECK:   moore.interface.signal @valid : !moore.l1
// CHECK: }

// Verify that the third instance (another_narrow_bus) with WIDTH=16 reuses the
// first interface and does NOT create a param_if_2.
// CHECK-NOT: moore.interface @param_if_2
