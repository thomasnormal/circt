// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test disable soft constraint support for randomization
// The 'disable soft' construct discards soft constraints on a variable
// See IEEE 1800-2017 Section 18.5.14.2 "Discarding soft constraints"

// Basic disable soft constraint
class test_disable_soft_basic;
  rand int b;

  // Soft constraints on b
  constraint c1 {
    soft b > 4;
    soft b < 12;
  }

  // Disable all soft constraints on b
  constraint c2 { disable soft b; }

  // New soft constraint after disabling
  constraint c3 { soft b == 20; }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_disable_soft_basic
// CHECK: moore.class.propertydecl @b : !moore.i32
// CHECK-SAME: rand_mode rand

// CHECK: moore.constraint.block @c1
// CHECK: moore.constraint.expr %{{.*}} : i1 soft
// CHECK: moore.constraint.expr %{{.*}} : i1 soft

// CHECK: moore.constraint.block @c2
// CHECK: moore.constraint.disable_soft %{{.*}} : !moore.i32

// CHECK: moore.constraint.block @c3
// CHECK: moore.constraint.expr %{{.*}} : i1 soft


// Disable soft with multiple variables
class test_disable_soft_multiple;
  rand int a, b;

  constraint c_soft {
    soft a == 10;
    soft b == 20;
  }

  // Disable soft on both variables
  constraint c_disable { disable soft a; disable soft b; }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_disable_soft_multiple
// CHECK: moore.constraint.block @c_soft
// CHECK: moore.constraint.expr %{{.*}} : i1 soft
// CHECK: moore.constraint.expr %{{.*}} : i1 soft

// CHECK: moore.constraint.block @c_disable
// CHECK: moore.constraint.disable_soft %{{.*}} : !moore.i32
// CHECK: moore.constraint.disable_soft %{{.*}} : !moore.i32


// Disable soft with new constraint in same block
class test_disable_soft_same_block;
  rand int value;

  constraint c_soft { soft value > 0; }

  // Disable old soft and apply new one in same block
  constraint c_override {
    disable soft value;
    soft value == 100;
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_disable_soft_same_block
// CHECK: moore.constraint.block @c_soft
// CHECK: moore.constraint.expr %{{.*}} : i1 soft

// CHECK: moore.constraint.block @c_override
// CHECK: moore.constraint.disable_soft %{{.*}} : !moore.i32
// CHECK: moore.constraint.expr %{{.*}} : i1 soft


// Disable soft in derived class pattern (common UVM pattern)
class base_packet;
  rand bit [7:0] addr;
  rand bit [7:0] data;

  constraint c_soft_defaults {
    soft addr == 8'h00;
    soft data == 8'hFF;
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @base_packet
// CHECK: moore.constraint.block @c_soft_defaults
// CHECK: moore.constraint.expr %{{.*}} : i1 soft
// CHECK: moore.constraint.expr %{{.*}} : i1 soft


class derived_packet extends base_packet;
  // Override parent's soft constraints
  constraint c_override_defaults {
    disable soft addr;
    disable soft data;
    soft addr == 8'h10;
    soft data == 8'h00;
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @derived_packet
// CHECK: moore.constraint.block @c_override_defaults
// CHECK: moore.constraint.disable_soft %{{.*}} : !moore.i8
// CHECK: moore.constraint.disable_soft %{{.*}} : !moore.i8
// CHECK: moore.constraint.expr %{{.*}} : i1 soft
// CHECK: moore.constraint.expr %{{.*}} : i1 soft


module top;
  initial begin
    automatic test_disable_soft_basic t1 = new();
    automatic test_disable_soft_multiple t2 = new();
    automatic test_disable_soft_same_block t3 = new();
    automatic base_packet t4 = new();
    automatic derived_packet t5 = new();

    void'(t1.randomize());
    void'(t2.randomize());
    void'(t3.randomize());
    void'(t4.randomize());
    void'(t5.randomize());
  end
endmodule
