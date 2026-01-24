// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s

// Test vtable inheritance for deep class hierarchies (3+ levels)
// This tests the scenario where a derived class overrides a method
// that was defined in a grandparent class (skipping the parent).

// CHECK: moore.class.classdecl @base_object
class base_object;
  // CHECK: moore.class.methoddecl @do_print -> @"base_object::do_print"
  virtual function void do_print();
    $display("base_object::do_print");
  endfunction
endclass

// CHECK: moore.class.classdecl @base_component extends @base_object
class base_component extends base_object;
  // Adds build_phase method
  // CHECK: moore.class.methoddecl @build_phase -> @"base_component::build_phase"
  virtual function void build_phase();
    $display("base_component::build_phase");
  endfunction
endclass

// CHECK: moore.class.classdecl @base_test extends @base_component
class base_test extends base_component;
  // Does NOT override build_phase - inherits from base_component
  // But has other methods
  // CHECK: moore.class.methoddecl @run_test -> @"base_test::run_test"
  virtual function void run_test();
    $display("base_test::run_test");
  endfunction
  // Inherited build_phase should point to base_component::build_phase
  // CHECK: moore.class.methoddecl @build_phase -> @"base_component::build_phase"
endclass

// CHECK: moore.class.classdecl @derived_test extends @base_test
class derived_test extends base_test;
  // DOES override build_phase (skipping base_test!)
  // CHECK: moore.class.methoddecl @build_phase -> @"derived_test::build_phase"
  virtual function void build_phase();
    $display("derived_test::build_phase");
  endfunction
  // Inherited run_test should point to base_test::run_test
  // CHECK: moore.class.methoddecl @run_test -> @"base_test::run_test"
endclass

// Verify vtable for derived_test has correct entries
// All build_phase entries in nested vtables point to the most derived impl
// CHECK: moore.vtable @derived_test::@vtable
// CHECK: moore.vtable @base_test::@vtable
// CHECK: moore.vtable @base_component::@vtable
// CHECK: moore.vtable @base_object::@vtable
// CHECK: moore.vtable_entry @do_print -> @"base_object::do_print"
// CHECK: moore.vtable_entry @build_phase -> @"derived_test::build_phase"
// CHECK: moore.vtable_entry @do_print -> @"base_object::do_print"
// CHECK: moore.vtable_entry @run_test -> @"base_test::run_test"
// CHECK: moore.vtable_entry @build_phase -> @"derived_test::build_phase"

module top;
  initial begin
    derived_test t;
    t = new();
    // Virtual dispatch should use vtable
    // CHECK: moore.vtable.load_method {{.*}} : @build_phase of <@derived_test>
    t.build_phase();  // Should call derived_test::build_phase
    // CHECK: moore.vtable.load_method {{.*}} : @do_print of <@derived_test>
    t.do_print();     // Should call base_object::do_print
    // CHECK: moore.vtable.load_method {{.*}} : @run_test of <@derived_test>
    t.run_test();     // Should call base_test::run_test
  end
endmodule
