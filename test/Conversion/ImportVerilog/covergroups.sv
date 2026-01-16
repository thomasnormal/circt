// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Test that covergroup definitions and instantiation are properly imported.

module CovergroupTest;
  logic clk;
  logic [7:0] data;
  logic [3:0] addr;

  // CHECK: moore.module @CovergroupTest
  // CHECK: moore.covergroup.inst @cg
  // CHECK: moore.covergroup.decl @cg
  // CHECK:   moore.coverpoint.decl @data_cp
  // CHECK:   moore.coverpoint.decl @addr_cp
  covergroup cg @(posedge clk);
    coverpoint data;
    coverpoint addr;
    // Cross coverage targets are not yet fully supported
    // cross data, addr;
  endgroup

  cg cg_inst = new();
endmodule
