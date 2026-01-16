// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test basic clocking block parsing and conversion to Moore dialect ops.
// See IEEE 1800-2017 Section 14 "Clocking Blocks".

module test_clocking_blocks;
  logic clk;
  logic [7:0] data_in;
  logic [7:0] data_out;
  logic ready, valid;

  // CHECK: moore.clocking_block @cb1 {
  // CHECK: }
  clocking cb1 @(posedge clk);
  endclocking

  // CHECK: moore.clocking_block @cb2 {
  // CHECK:   moore.clocking_block.signal @data_in : !moore.l8
  // CHECK: }
  clocking cb2 @(posedge clk);
    input data_in;
  endclocking

  // CHECK: moore.clocking_block @cb3 {
  // CHECK:   moore.clocking_block.signal @data_out : !moore.l8
  // CHECK: }
  clocking cb3 @(posedge clk);
    output data_out;
  endclocking

  // CHECK: moore.clocking_block @cb4 {
  // CHECK:   moore.clocking_block.signal @data_in : !moore.l8
  // CHECK:   moore.clocking_block.signal @data_out : !moore.l8
  // CHECK: }
  clocking cb4 @(posedge clk);
    input data_in;
    output data_out;
  endclocking

  // CHECK: moore.clocking_block @cb5 {
  // CHECK:   moore.clocking_block.signal @ready : !moore.l1
  // CHECK:   moore.clocking_block.signal @valid : !moore.l1
  // CHECK: }
  clocking cb5 @(posedge clk);
    input ready;
    output valid;
  endclocking

  // Clocking block with default skews
  // CHECK: moore.clocking_block @cb_skew {
  // CHECK: }
  clocking cb_skew @(posedge clk);
    default input #1step output #0;
  endclocking

endmodule
