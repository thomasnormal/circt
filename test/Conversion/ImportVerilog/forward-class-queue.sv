// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test that variables with forward-declared class types can be converted
// before the class body is defined. This tests the two-pass conversion
// approach where class declarations are created during type conversion
// but bodies are populated later.

package test_pkg;
  typedef class my_class;
  my_class items[$];  // queue of forward-declared class

  class my_class;
    int value;
  endclass
endpackage

// CHECK-DAG: moore.global_variable @"test_pkg::items" : !moore.queue<class<@"test_pkg::my_class">, 0>
// CHECK-DAG: moore.class.classdecl @"test_pkg::my_class"
// CHECK: moore.class.propertydecl @value : !moore.i32
