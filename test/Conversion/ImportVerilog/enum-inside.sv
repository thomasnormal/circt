// RUN: circt-verilog --ir-moore %s | FileCheck %s

// CHECK-LABEL: moore.class.classdecl @enum_inside {
// CHECK:   moore.class.propertydecl @val : !moore.i32 rand_mode rand
// CHECK:   moore.constraint.block @c_enum_inside {
// CHECK:     moore.constraint.expr {{.*}} : i1
// CHECK:   }
// CHECK: }
class enum_inside;
  rand int val;
  typedef enum {ON, OFF} vals_e;

  // Use enum members directly in inside constraint
  constraint c_enum_inside {
    val inside {ON, OFF};
  }

  function new();
  endfunction
endclass
