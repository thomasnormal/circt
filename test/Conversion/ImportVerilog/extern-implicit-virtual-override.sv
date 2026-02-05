// RUN: circt-verilog --ir-hw %s | FileCheck %s
// REQUIRES: slang

// Test implicit virtual override when extern prototype omits 'virtual'.
// Base declares virtual method. Derived extern prototype should still be virtual.

package implicit_pkg;
  class base_class;
    virtual function string get_name();
      return "base";
    endfunction
  endclass
endpackage

class derived_class extends implicit_pkg::base_class;
  // No 'virtual' keyword here, but should still be virtual due to override.
  extern function string get_name();
endclass

function string derived_class::get_name();
  return "derived";
endfunction

// CHECK-DAG: llvm.mlir.global internal @"derived_class::__vtable__"{{.*}}circt.vtable_entries = [{{.*}}@"derived_class::get_name"
// CHECK-DAG: llvm.mlir.global internal @"implicit_pkg::base_class::__vtable__"{{.*}}circt.vtable_entries
