// RUN: circt-verilog --no-uvm-auto-include %s --ir-hw | FileCheck %s

// Test that parameterized class specializations have their static member
// initializers properly generated. This is critical for UVM factory
// registration patterns where static member initializers register types.

package test_pkg;
  // Simulates uvm_registry_common pattern
  class registry_common #(type T, string Tname);
    // Static member with initializer - should be generated for each specialization
    static bit m__initialized = __deferred_init();
    
    static function bit __deferred_init();
      $display("Registering %s", Tname);
      return 1;
    endfunction
  endclass
  
  // Base class
  class base_component;
    function new();
    endfunction
  endclass
  
  // User class that triggers specialization
  class my_test extends base_component;
    typedef registry_common#(my_test, "my_test") type_id;
    
    function new();
      super.new();
    endfunction
  endclass

  class another_test extends base_component;
    typedef registry_common#(another_test, "another_test") type_id;
    
    function new();
      super.new();
    endfunction
  endclass
endpackage

module top;
  import test_pkg::*;
  
  initial begin
    my_test t1;
    another_test t2;
    t1 = new();
    t2 = new();
    $display("Tests created");
  end
endmodule

// Verify that both specializations have their static members generated:
// CHECK: llvm.mlir.global internal @"test_pkg::test_pkg::registry_common::m__initialized"
// CHECK: llvm.mlir.global_ctors ctors = [@"__moore_global_init_test_pkg::test_pkg::registry_common::m__initialized"]
// CHECK: llvm.func internal @"__moore_global_init_test_pkg::test_pkg::registry_common::m__initialized"

// CHECK: llvm.mlir.global internal @"test_pkg::test_pkg::registry_common_0::m__initialized"
// CHECK: llvm.mlir.global_ctors ctors = [@"__moore_global_init_test_pkg::test_pkg::registry_common_0::m__initialized"]
// CHECK: llvm.func internal @"__moore_global_init_test_pkg::test_pkg::registry_common_0::m__initialized"
