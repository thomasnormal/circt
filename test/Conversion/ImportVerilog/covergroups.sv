// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Test that covergroup definitions and instantiation are properly imported.

module CovergroupTest;
  logic clk;
  logic [7:0] data;
  logic [3:0] addr;

  // CHECK: moore.module @CovergroupTest
  // CHECK: moore.covergroup.decl @cg
  // CHECK:   moore.coverpoint.decl @data
  // CHECK:   moore.coverpoint.decl @addr
  // CHECK:   moore.covercross.decl @data_X_addr
  covergroup cg @(posedge clk);
    coverpoint data;
    coverpoint addr;
    cross data, addr;
  endgroup

  // CHECK: moore.covergroup.inst @cg
  cg cg_inst = new();
endmodule
