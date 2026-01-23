// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test that logical AND/OR operations correctly handle mixed bit/logic types.
// When one operand is 2-state (bit) and another is 4-state (logic), both should
// be promoted to 4-state for the logical operation.

// CHECK-LABEL: @BitLogicMix
module BitLogicMix;
  bit a;       // 2-state
  logic b;     // 4-state
  logic result;

  // Logical AND with mixed types should convert i1 to l1
  // CHECK: moore.int_to_logic %{{.*}} : i1
  // CHECK-NEXT: moore.and %{{.*}}, %{{.*}} : l1
  initial result = a && b;
endmodule

// CHECK-LABEL: @BitBoolMix
module BitBoolMix;
  bit a;
  bit b;
  bit result;

  // Both 2-state should remain i1
  // CHECK: moore.and %{{.*}}, %{{.*}} : i1
  initial result = a && b;
endmodule

// CHECK-LABEL: @LogicLogicMix
module LogicLogicMix;
  logic a;
  logic b;
  logic result;

  // Both 4-state should remain l1
  // CHECK: moore.and %{{.*}}, %{{.*}} : l1
  initial result = a && b;
endmodule

// CHECK-LABEL: @ComplexMix
module ComplexMix;
  bit a;
  logic b;
  logic c;
  logic d;

  // Complex expression with mixed types
  // The comparison (a == 1) produces i1 (2-state)
  // The b && c produces l1 (4-state)
  // The outer && should convert i1 to l1
  initial d = (a == 1'b1) && (b && c);
endmodule
