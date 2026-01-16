// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test basic clocking block parsing and conversion to Moore dialect ops.
// See IEEE 1800-2017 Section 14 "Clocking Blocks".

module test_clocking_blocks;
  logic clk;
  logic [7:0] data_in;
  logic [7:0] data_out;
  logic ready, valid;

  // CHECK: moore.clocking_block "cb1" {
  // CHECK: }
  clocking cb1 @(posedge clk);
  endclocking

  // CHECK: moore.clocking_block "cb2" {
  // CHECK:   moore.clocking_block.signal "data_in" : !moore.l8
  // CHECK: }
  clocking cb2 @(posedge clk);
    input data_in;
  endclocking

  // CHECK: moore.clocking_block "cb3" {
  // CHECK:   moore.clocking_block.signal "data_out" : !moore.l8
  // CHECK: }
  clocking cb3 @(posedge clk);
    output data_out;
  endclocking

  // CHECK: moore.clocking_block "cb4" {
  // CHECK:   moore.clocking_block.signal "data_in" : !moore.l8
  // CHECK:   moore.clocking_block.signal "data_out" : !moore.l8
  // CHECK: }
  clocking cb4 @(posedge clk);
    input data_in;
    output data_out;
  endclocking

  // CHECK: moore.clocking_block "cb5" {
  // CHECK:   moore.clocking_block.signal "ready" : !moore.l1
  // CHECK:   moore.clocking_block.signal "valid" : !moore.l1
  // CHECK: }
  clocking cb5 @(posedge clk);
    input ready;
    output valid;
  endclocking

  // Clocking block with default skews
  // CHECK: moore.clocking_block "cb_skew" {
  // CHECK: }
  clocking cb_skew @(posedge clk);
    default input #1step output #0;
  endclocking

endmodule

//===----------------------------------------------------------------------===//
// Additional tests from sv-tests Chapter 14
//===----------------------------------------------------------------------===//

// Test basic clocking block (14.3--clocking-block.sv)
// CHECK-LABEL: moore.module @test_basic_clocking
module test_basic_clocking(input clk);
  // CHECK: moore.clocking_block "ck1" {
  // CHECK: }
  clocking ck1 @(posedge clk);
    default input #10ns output #5ns;
  endclocking
endmodule

// Test clocking block with signals (14.3--clocking-block-signals.sv)
// CHECK-LABEL: moore.module @test_signals_clocking
module test_signals_clocking(input clk, input a, output logic b, output logic c);
  // CHECK: moore.clocking_block "ck1" {
  // CHECK:   moore.clocking_block.signal "a" : !moore.l1
  // CHECK:   moore.clocking_block.signal "b" : !moore.l1
  // CHECK:   moore.clocking_block.signal "c" : !moore.l1
  // CHECK: }
  clocking ck1 @(posedge clk);
    default input #10ns output #5ns;
    input a;
    output b;
    output #3ns c;  // Per-signal skew override
  endclocking

  always_ff @(posedge clk) begin
    b <= a;
    c <= a;
  end
endmodule

// Test default clocking block (14.3--default-clocking-block.sv)
// CHECK-LABEL: moore.module @test_default_clocking
module test_default_clocking(input clk);
  // CHECK: moore.clocking_block "" {
  // CHECK: }
  default clocking @(posedge clk);
    default input #10ns output #5ns;
  endclocking
endmodule

// Test global clocking block (14.3--global-clocking-block.sv)
// CHECK-LABEL: moore.module @test_global_clocking
module test_global_clocking(input clk);
  // CHECK: moore.clocking_block "ck1" {
  // CHECK: }
  global clocking ck1 @(posedge clk); endclocking
endmodule

//===----------------------------------------------------------------------===//
// Additional edge cases and features
//===----------------------------------------------------------------------===//

// Test clocking block with negedge
// CHECK-LABEL: moore.module @test_negedge_clocking
module test_negedge_clocking(input clk, input logic [7:0] data);
  // CHECK: moore.clocking_block "cb_neg" {
  // CHECK:   moore.clocking_block.signal "data" : !moore.l8
  // CHECK: }
  clocking cb_neg @(negedge clk);
    input data;
  endclocking
endmodule

// Test clocking block with inout signals
// CHECK-LABEL: moore.module @test_inout_clocking
module test_inout_clocking(input clk, input logic [7:0] data_in, output logic [7:0] data_out);
  logic [7:0] bidirectional;

  // CHECK: moore.clocking_block "cb" {
  // CHECK:   moore.clocking_block.signal "data_in" : !moore.l8
  // CHECK:   moore.clocking_block.signal "data_out" : !moore.l8
  // CHECK:   moore.clocking_block.signal "bidirectional" : !moore.l8
  // CHECK: }
  clocking cb @(posedge clk);
    input data_in;
    output data_out;
    inout bidirectional;
  endclocking
endmodule

// Test clocking block with array signals
// CHECK-LABEL: moore.module @test_array_clocking
module test_array_clocking(input clk, input logic [7:0] mem [0:15]);
  // CHECK: moore.clocking_block "cb" {
  // CHECK:   moore.clocking_block.signal "mem" : !moore.uarray<16 x l8>
  // CHECK: }
  clocking cb @(posedge clk);
    input mem;
  endclocking
endmodule

// Test clocking block with struct signals
typedef struct packed {
  logic [7:0] data;
  logic valid;
  logic ready;
} bus_data_t;

// CHECK-LABEL: moore.module @test_struct_clocking
module test_struct_clocking(input clk, input bus_data_t bus);
  // CHECK: moore.clocking_block "cb" {
  // CHECK:   moore.clocking_block.signal "bus" : !moore.struct<{data: l8, valid: l1, ready: l1}>
  // CHECK: }
  clocking cb @(posedge clk);
    input bus;
  endclocking
endmodule

// Test multiple clocking blocks in same module
// CHECK-LABEL: moore.module @test_multiple_clocking
module test_multiple_clocking(input clk1, input clk2, input logic [7:0] data);
  // CHECK: moore.clocking_block "cb1" {
  // CHECK:   moore.clocking_block.signal "data" : !moore.l8
  // CHECK: }
  clocking cb1 @(posedge clk1);
    input data;
  endclocking

  // CHECK: moore.clocking_block "cb2" {
  // CHECK:   moore.clocking_block.signal "data" : !moore.l8
  // CHECK: }
  clocking cb2 @(negedge clk2);
    input data;
  endclocking
endmodule
