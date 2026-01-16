// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test clocking block signal access through cb.signal syntax.
// This tests the rvalue and lvalue generation for ClockVar expressions.
// See IEEE 1800-2017 Section 14 "Clocking Blocks".

//===----------------------------------------------------------------------===//
// Basic clocking block signal read (rvalue)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @test_cb_read
module test_cb_read(input clk, input logic [7:0] data_in, output logic [7:0] result);
  // CHECK: moore.clocking_block "cb" {
  // CHECK:   moore.clocking_block.signal "data_in" : !moore.l8
  // CHECK: }
  clocking cb @(posedge clk);
    input data_in;
  endclocking

  // Reading cb.data_in should resolve to reading data_in
  // CHECK: moore.read
  // CHECK: moore.blocking_assign
  always_comb begin
    result = cb.data_in;
  end
endmodule

//===----------------------------------------------------------------------===//
// Basic clocking block signal write (lvalue) - must use non-blocking assign
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @test_cb_write
module test_cb_write(input clk, input logic [7:0] data_in, output logic [7:0] data_out);
  // CHECK: moore.clocking_block "cb" {
  // CHECK:   moore.clocking_block.signal "data_out" : !moore.l8
  // CHECK: }
  clocking cb @(posedge clk);
    output data_out;
  endclocking

  // Writing to cb.data_out should resolve to writing to data_out
  // Per IEEE 1800-2017, clocking outputs must use non-blocking assignment
  // CHECK: moore.read
  // CHECK: moore.nonblocking_assign
  always_ff @(posedge clk) begin
    cb.data_out <= data_in;
  end
endmodule

//===----------------------------------------------------------------------===//
// Clocking block with both input and output signals
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @test_cb_inout
module test_cb_inout(input clk, input logic [7:0] a, output logic [7:0] b);
  // CHECK: moore.clocking_block "cb" {
  // CHECK:   moore.clocking_block.signal "a" : !moore.l8
  // CHECK:   moore.clocking_block.signal "b" : !moore.l8
  // CHECK: }
  clocking cb @(posedge clk);
    input a;
    output b;
  endclocking

  // Use both input and output signals through clocking block
  // CHECK: moore.read
  // CHECK: moore.nonblocking_assign
  always_ff @(posedge clk) begin
    cb.b <= cb.a;
  end
endmodule

//===----------------------------------------------------------------------===//
// Clocking block signal access in always_ff
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @test_cb_always_ff
module test_cb_always_ff(input clk, input logic [7:0] data_in, output logic [7:0] data_out);
  // CHECK: moore.clocking_block "cb" {
  // CHECK:   moore.clocking_block.signal "data_in" : !moore.l8
  // CHECK:   moore.clocking_block.signal "data_out" : !moore.l8
  // CHECK: }
  clocking cb @(posedge clk);
    input data_in;
    output data_out;
  endclocking

  // CHECK: moore.read
  // CHECK: moore.nonblocking_assign
  always_ff @(posedge clk) begin
    cb.data_out <= cb.data_in;
  end
endmodule

//===----------------------------------------------------------------------===//
// Clocking block with inout signal
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @test_cb_inout_signal
module test_cb_inout_signal(input clk, inout logic [7:0] bidir);
  // CHECK: moore.clocking_block "cb" {
  // CHECK:   moore.clocking_block.signal "bidir" : !moore.l8
  // CHECK: }
  clocking cb @(posedge clk);
    inout bidir;
  endclocking

  logic [7:0] temp;
  // Read through clocking block
  // CHECK: moore.read
  // CHECK: moore.blocking_assign
  always_comb begin
    temp = cb.bidir;
  end
endmodule

//===----------------------------------------------------------------------===//
// Clocking block signal access with expressions
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @test_cb_expressions
module test_cb_expressions(input clk, input logic [7:0] a, input logic [7:0] b, output logic [7:0] result);
  // CHECK: moore.clocking_block "cb" {
  // CHECK:   moore.clocking_block.signal "a" : !moore.l8
  // CHECK:   moore.clocking_block.signal "b" : !moore.l8
  // CHECK:   moore.clocking_block.signal "result" : !moore.l8
  // CHECK: }
  clocking cb @(posedge clk);
    input a;
    input b;
    output result;
  endclocking

  // Use clocking block signals in expressions
  // CHECK: moore.read
  // CHECK: moore.read
  // CHECK: moore.add
  // CHECK: moore.nonblocking_assign
  always_ff @(posedge clk) begin
    cb.result <= cb.a + cb.b;
  end
endmodule

//===----------------------------------------------------------------------===//
// Multiple clocking blocks accessing same signals
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @test_multiple_cb
module test_multiple_cb(input clk1, input clk2, input logic [7:0] data);
  logic [7:0] temp1, temp2;

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

  // CHECK: moore.read
  // CHECK: moore.blocking_assign
  // CHECK: moore.read
  // CHECK: moore.blocking_assign
  always_comb begin
    temp1 = cb1.data;
    temp2 = cb2.data;
  end
endmodule

//===----------------------------------------------------------------------===//
// Clocking block with struct signals
//===----------------------------------------------------------------------===//

typedef struct packed {
  logic [7:0] data;
  logic valid;
} packet_t;

// CHECK-LABEL: moore.module @test_cb_struct
module test_cb_struct(input clk, input packet_t pkt_in, output packet_t pkt_out);
  // CHECK: moore.clocking_block "cb" {
  // CHECK:   moore.clocking_block.signal "pkt_in"
  // CHECK:   moore.clocking_block.signal "pkt_out"
  // CHECK: }
  clocking cb @(posedge clk);
    input pkt_in;
    output pkt_out;
  endclocking

  // CHECK: moore.read
  // CHECK: moore.nonblocking_assign
  always_ff @(posedge clk) begin
    cb.pkt_out <= cb.pkt_in;
  end
endmodule
