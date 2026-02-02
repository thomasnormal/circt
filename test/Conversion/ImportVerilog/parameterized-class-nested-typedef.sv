// RUN: circt-verilog %s --ir-hw 2>&1 | FileCheck %s

// Test that parameterized class specializations referenced via nested typedefs
// inside classes (like UVM's `uvm_component_utils macro) are properly converted
// with separate static members for each specialization.

// Parameterized registry class (similar to uvm_component_registry)
class registry #(type T=int, string Tname="T");
  // Static counter - each specialization should get its own
  static int counter = 0;

  static function int get_id();
    counter++;
    return counter;
  endfunction
endclass

// First user class with nested typedef
class class_a;
  typedef registry #(class_a, "class_a") type_id;

  static function type_id get_type();
    return null;
  endfunction
endclass

// Second user class with nested typedef - should get separate registry
class class_b;
  typedef registry #(class_b, "class_b") type_id;

  static function type_id get_type();
    return null;
  endfunction
endclass

// Module-level typedef - also gets its own specialization
typedef registry #(int, "int_spec") int_registry;

module top;
  initial begin
    // Access static members from all three specializations
    $display("class_a counter = %d", class_a::type_id::counter);
    $display("class_b counter = %d", class_b::type_id::counter);
    $display("int counter = %d", int_registry::counter);
  end
endmodule

// There should be three separate specializations with their own counters.
// The base specialization always gets "registry", others get numbered.
// CHECK: llvm.mlir.global internal @"registry::counter"
// CHECK: llvm.mlir.global internal @"registry_{{[0-9]+}}::counter"
// CHECK: llvm.mlir.global internal @"registry_{{[0-9]+}}::counter"
