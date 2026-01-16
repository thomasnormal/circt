// RUN: circt-opt --verify-diagnostics --verify-roundtrip %s | FileCheck %s

// Test basic covergroup declaration
// CHECK-LABEL: moore.covergroup.decl @empty_cg {
// CHECK-NEXT: }
moore.covergroup.decl @empty_cg {
}

// Test covergroup with coverpoint declarations
// CHECK-LABEL: moore.covergroup.decl @cg_with_coverpoints {
// CHECK-NEXT:   moore.coverpoint.decl @state_cp : i4
// CHECK-NEXT:   moore.coverpoint.decl @data_cp : i8
// CHECK-NEXT: }
moore.covergroup.decl @cg_with_coverpoints {
  moore.coverpoint.decl @state_cp : i4
  moore.coverpoint.decl @data_cp : i8
}

// Test covergroup with cross coverage declaration
// CHECK-LABEL: moore.covergroup.decl @cg_with_cross {
// CHECK-NEXT:   moore.coverpoint.decl @x_cp : i4
// CHECK-NEXT:   moore.coverpoint.decl @y_cp : i4
// CHECK-NEXT:   moore.covercross.decl @xy_cross targets [@x_cp, @y_cp]
// CHECK-NEXT: }
moore.covergroup.decl @cg_with_cross {
  moore.coverpoint.decl @x_cp : i4
  moore.coverpoint.decl @y_cp : i4
  moore.covercross.decl @xy_cross targets [@x_cp, @y_cp]
}

// Test multiple covergroups
// CHECK-LABEL: moore.covergroup.decl @cg1 {
moore.covergroup.decl @cg1 {
  moore.coverpoint.decl @cp1 : i16
}

// CHECK-LABEL: moore.covergroup.decl @cg2 {
moore.covergroup.decl @cg2 {
  moore.coverpoint.decl @cp2 : i32
  moore.covercross.decl @cross1 targets [@cp2]
}
