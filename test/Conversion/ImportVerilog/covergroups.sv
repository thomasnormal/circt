// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Test that covergroup definitions are skipped with a remark, not a hard error.
// Covergroups are not yet fully supported but should not block UVM code import.

module CovergroupTest;
  logic clk;
  logic [7:0] data;
  logic [3:0] addr;

  // CHECK: moore.module @CovergroupTest
  covergroup cg @(posedge clk);
    coverpoint data;
    coverpoint addr;
    cross data, addr;
  endgroup

  // Note: covergroup instantiation (cg cg_inst = new();) is not yet supported.
  // This test verifies that the covergroup definition itself doesn't cause errors.
endmodule
