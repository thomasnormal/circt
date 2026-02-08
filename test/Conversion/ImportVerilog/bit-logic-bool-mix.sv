// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s | FileCheck %s

// Test that logical AND operations correctly handle mixed bit/logic types
// with short-circuit evaluation (IEEE 1800-2017 §11.4.7).
// When one operand is 2-state (bit) and another is 4-state (logic), both should
// be promoted to 4-state. The && lowers to moore.conditional for short-circuit.

// CHECK-LABEL: @BitLogicMix
module BitLogicMix;
  bit a;       // 2-state
  logic b;     // 4-state
  logic result;

  // Logical AND with mixed types: LHS i1 promoted to l1, short-circuit
  // CHECK: moore.int_to_logic %{{.*}} : i1
  // CHECK-NEXT: moore.conditional %{{.*}} : l1 -> l1
  initial result = a && b;
endmodule

// CHECK-LABEL: @BitBoolMix
module BitBoolMix;
  bit a;
  bit b;
  bit result;

  // Both 2-state: short-circuit conditional on i1
  // CHECK: moore.conditional %{{.*}} : i1 -> i1
  initial result = a && b;
endmodule

// CHECK-LABEL: @LogicLogicMix
module LogicLogicMix;
  logic a;
  logic b;
  logic result;

  // Both 4-state: short-circuit conditional on l1
  // CHECK: moore.conditional %{{.*}} : l1 -> l1
  initial result = a && b;
endmodule

// CHECK-LABEL: @ComplexMix
module ComplexMix;
  bit a;
  logic b;
  logic c;
  logic d;

  // Complex expression with mixed types and nested short-circuit
  // CHECK: moore.conditional %{{.*}} : l1 -> l1
  initial d = (a == 1'b1) && (b && c);
endmodule
