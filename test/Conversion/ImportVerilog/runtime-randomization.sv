// RUN: circt-verilog %s --ir-moore | FileCheck %s
// RUN: circt-verilog %s --ir-llhd | FileCheck %s --check-prefix=CHECK-LOWERED
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Runtime Randomization Tests
//===----------------------------------------------------------------------===//
//
// This test verifies the compilation of SystemVerilog randomization constructs
// that rely on the MooreRuntime randomization functions:
// - __moore_randomize_basic()
// - __moore_randomize_with_range()
// - __moore_randc_next()
//

// CHECK-LOWERED-DAG: llvm.func @__moore_randomize_basic
// CHECK-LOWERED-DAG: llvm.func @__moore_randc_next

/// Test basic packet class with rand properties for runtime randomization
// CHECK-LABEL: moore.class.classdecl @packet {
// CHECK:   moore.class.propertydecl @addr : !moore.i8 rand_mode rand
// CHECK:   moore.class.propertydecl @data : !moore.i8 rand_mode rand
// CHECK: }

class packet;
  rand bit [7:0] addr;
  rand bit [7:0] data;

  function new();
    addr = 0;
    data = 0;
  endfunction
endclass

/// Test module that exercises randomize() on class instance
// CHECK-LABEL: moore.module @test() {
// CHECK:   moore.procedure initial {
// CHECK:     moore.class.new : <@packet>
// CHECK:     moore.randomize
// CHECK:   }
// CHECK: }

// CHECK-LOWERED: llvm.call @__moore_randomize_basic

module test;
  initial begin
    packet p = new();
    int success;
    success = p.randomize();
  end
endmodule

//===----------------------------------------------------------------------===//
// Extended randomization tests
//===----------------------------------------------------------------------===//

/// Test class with randc (cyclic randomization) property
// CHECK-LABEL: moore.class.classdecl @packet_randc {
// CHECK:   moore.class.propertydecl @id : !moore.i4 rand_mode randc
// CHECK:   moore.class.propertydecl @data : !moore.i8 rand_mode rand
// CHECK: }

class packet_randc;
  randc bit [3:0] id;  // Cyclic - must visit all values before repeating
  rand bit [7:0] data;

  function new();
    id = 0;
    data = 0;
  endfunction
endclass

// CHECK-LABEL: moore.module @test_randc() {
// CHECK:   moore.procedure initial {
// CHECK:     moore.class.new : <@packet_randc>
// CHECK:     moore.randomize
// CHECK:   }
// CHECK: }

// CHECK-LOWERED: llvm.call @__moore_randc_next

module test_randc;
  initial begin
    packet_randc p = new();
    void'(p.randomize());
    void'(p.randomize());
  end
endmodule

/// Test class with constraint block
// CHECK-LABEL: moore.class.classdecl @packet_constrained {
// CHECK:   moore.class.propertydecl @addr : !moore.i8 rand_mode rand
// CHECK:   moore.constraint.block @valid_addr {
// CHECK:   }
// CHECK: }

class packet_constrained;
  rand bit [7:0] addr;
  constraint valid_addr { addr >= 10; addr <= 100; }

  function new();
    addr = 0;
  endfunction
endclass

// CHECK-LABEL: moore.module @test_constrained() {

module test_constrained;
  initial begin
    packet_constrained p = new();
    void'(p.randomize());
  end
endmodule

/// Test multiple rand properties
// CHECK-LABEL: moore.class.classdecl @multi_packet {
// CHECK:   moore.class.propertydecl @a : !moore.i16 rand_mode rand
// CHECK:   moore.class.propertydecl @b : !moore.i16 rand_mode rand
// CHECK:   moore.class.propertydecl @c : !moore.i16 rand_mode rand
// CHECK: }

class multi_packet;
  rand bit [15:0] a;
  rand bit [15:0] b;
  rand bit [15:0] c;

  function new();
    a = 0;
    b = 0;
    c = 0;
  endfunction
endclass

// CHECK-LABEL: moore.module @test_multi() {

module test_multi;
  initial begin
    multi_packet p = new();
    int success;
    success = p.randomize();
    $display("a=%0d b=%0d c=%0d success=%0d", p.a, p.b, p.c, success);
  end
endmodule

/// Test randomize() in loop
// CHECK-LABEL: moore.module @test_loop() {

module test_loop;
  initial begin
    packet p = new();
    for (int i = 0; i < 10; i++) begin
      void'(p.randomize());
    end
  end
endmodule
