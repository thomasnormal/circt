// RUN: circt-verilog --ir-hw %s | FileCheck %s

// Test that extern virtual methods properly generate vtable entries.
// The key issue was that when a method is declared as "extern virtual" in
// a class, the out-of-class implementation doesn't have the Virtual flag
// in slang - only the prototype does. This test verifies that the vtable
// generation uses the prototype's flag.

package test_pkg;
  class base_class;
    virtual function string get_name();
      return "base_class";
    endfunction

    virtual function int get_value();
      return 0;
    endfunction
  endclass
endpackage

// Derived class with extern virtual method declarations
class derived_class extends test_pkg::base_class;
  extern virtual function string get_name();
  extern virtual function int get_value();
  extern virtual function void do_something(int x);
endclass

// Out-of-class implementations (these don't have Virtual flag in slang)
function string derived_class::get_name();
  return "derived_class";
endfunction

function int derived_class::get_value();
  return 42;
endfunction

function void derived_class::do_something(int x);
  $display("do_something: %d", x);
endfunction

// Another derived class to test inheritance chain
class grandchild_class extends derived_class;
  extern virtual function string get_name();
endclass

function string grandchild_class::get_name();
  return "grandchild_class";
endfunction

// Vtables are emitted (grandchild first, then derived, then base)
// The format is [[index, @func], ...] where index is the vtable slot
// Each vtable should have the appropriate methods (order may vary by index)
// CHECK-DAG: llvm.mlir.global internal @"grandchild_class::__vtable__"{{.*}}circt.vtable_entries = [{{.*}}@"grandchild_class::get_name"
// CHECK-DAG: llvm.mlir.global internal @"derived_class::__vtable__"{{.*}}circt.vtable_entries = [{{.*}}@"derived_class::get_name"
// CHECK-DAG: llvm.mlir.global internal @"test_pkg::base_class::__vtable__"{{.*}}circt.vtable_entries
