// RUN: circt-verilog %s | FileCheck %s

// CHECK-LABEL: moore.class.classdecl @enum_inside {
// CHECK:   moore.class.propertydecl @val : !moore.i32 rand_mode rand
// CHECK:   moore.constraint.block @c_enum_inside {
// CHECK:     moore.constraint.expr {{.*}} : i1
// CHECK:   }
// CHECK: }
class enum_inside;
  rand int val;
  typedef enum {ON, OFF} vals_e;

  constraint c_enum_inside {
    val inside {vals_e};
  }

  function new();
  endfunction
endclass
