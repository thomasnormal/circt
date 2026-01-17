// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test soft constraint support for randomization
// Soft constraints provide default values that can be overridden by hard constraints
// See IEEE 1800-2017 Section 18.5.13 "Soft constraints"

// Basic soft constraint
class test_soft_basic;
  rand bit [7:0] value;
  rand int count;

  // Soft constraint - provides default but can be overridden
  constraint c_soft_default { soft value == 50; }

  // Hard constraint - always enforced
  constraint c_hard_range { count > 0; count < 100; }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_soft_basic
// CHECK: moore.class.propertydecl @value : !moore.i8
// CHECK-SAME: rand_mode rand
// CHECK: moore.class.propertydecl @count : !moore.i32
// CHECK-SAME: rand_mode rand

// CHECK: moore.constraint.block @c_soft_default
// CHECK: moore.constraint.expr %{{.*}} : i1 soft

// CHECK: moore.constraint.block @c_hard_range
// CHECK: moore.constraint.expr %{{.*}} : i1{{$}}
// CHECK: moore.constraint.expr %{{.*}} : i1{{$}}

// Multiple soft constraints in one block
class test_soft_multiple;
  rand int a, b, c;

  constraint c_defaults {
    soft a == 10;
    soft b == 20;
    soft c == a + b;  // Soft with expression
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_soft_multiple
// CHECK: moore.constraint.block @c_defaults
// CHECK: moore.constraint.expr %{{.*}} : i1 soft
// CHECK: moore.constraint.expr %{{.*}} : i1 soft
// CHECK: moore.constraint.expr %{{.*}} : i1 soft

// Mixed hard and soft constraints
class test_soft_mixed;
  rand bit [15:0] addr;
  rand bit [7:0] data;

  constraint c_mixed {
    addr inside {[16'h1000:16'h2000]};  // Hard constraint
    soft data == 8'hFF;                  // Soft default
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_soft_mixed
// CHECK: moore.constraint.block @c_mixed
// CHECK: moore.constraint.expr %{{.*}} : i1{{$}}
// CHECK: moore.constraint.expr %{{.*}} : i1 soft

// Soft constraint with conditional
class test_soft_conditional;
  rand bit mode;
  rand int value;

  constraint c_cond {
    soft mode == 1;
    if (mode) {
      soft value > 0;
    } else {
      soft value < 0;
    }
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_soft_conditional
// CHECK: moore.constraint.block @c_cond
// CHECK: moore.constraint.expr %{{.*}} : i1 soft
// CHECK: moore.constraint.if_else
// CHECK: moore.constraint.expr %{{.*}} : i1 soft
// CHECK: } else {
// CHECK: moore.constraint.expr %{{.*}} : i1 soft
// CHECK: }

// Soft constraint with implication
class test_soft_implication;
  rand bit enable;
  rand int threshold;

  constraint c_impl {
    enable -> soft threshold == 100;  // Soft in implication
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_soft_implication
// CHECK: moore.constraint.block @c_impl
// CHECK: moore.constraint.implication
// CHECK: moore.constraint.expr %{{.*}} : i1 soft

// Soft constraint in foreach
class test_soft_foreach;
  rand int arr[4];

  constraint c_foreach {
    foreach (arr[i]) {
      soft arr[i] == i * 10;  // Soft default for each element
    }
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_soft_foreach
// CHECK: moore.constraint.block @c_foreach
// CHECK: moore.constraint.foreach %{{.*}} : !moore.uarray<4 x i32>
// CHECK-NEXT: ^bb0(%{{.*}}: !moore.i32):
// CHECK: moore.constraint.expr %{{.*}} : i1 soft

// UVM-style soft constraint pattern
class uvm_style_soft;
  rand bit [31:0] address;
  rand bit [7:0] burst_len;
  rand bit [2:0] burst_type;

  // Soft defaults for common values, can be overridden in tests
  constraint c_defaults {
    soft address == 32'h0;
    soft burst_len == 1;
    soft burst_type == 0;
  }

  // Hard constraints for protocol compliance
  constraint c_protocol {
    burst_len inside {[1:256]};
    burst_type inside {[0:7]};
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @uvm_style_soft
// CHECK: moore.constraint.block @c_defaults
// CHECK: moore.constraint.expr %{{.*}} : i1 soft
// CHECK: moore.constraint.expr %{{.*}} : i1 soft
// CHECK: moore.constraint.expr %{{.*}} : i1 soft

// CHECK: moore.constraint.block @c_protocol
// CHECK: moore.constraint.expr %{{.*}} : i1{{$}}
// CHECK: moore.constraint.expr %{{.*}} : i1{{$}}

module top;
  initial begin
    automatic test_soft_basic t1 = new();
    automatic test_soft_multiple t2 = new();
    automatic test_soft_mixed t3 = new();
    automatic test_soft_conditional t4 = new();
    automatic test_soft_implication t5 = new();
    automatic test_soft_foreach t6 = new();
    automatic uvm_style_soft t7 = new();

    void'(t1.randomize());
    void'(t2.randomize());
    void'(t3.randomize());
    void'(t4.randomize());
    void'(t5.randomize());
    void'(t6.randomize());
    void'(t7.randomize());
  end
endmodule
