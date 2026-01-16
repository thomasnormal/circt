// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Test coverage collection with explicit bins.
// This tests the infrastructure for SystemVerilog covergroups with named bins.

module test_coverage;
  logic clk;
  logic [3:0] data;

  // CHECK: moore.module @test_coverage
  // CHECK: moore.covergroup.decl @cg
  // CHECK:   moore.coverpoint.decl @cp_data

  // Basic covergroup with a coverpoint
  // Note: Explicit bin definitions are not yet fully supported in the import,
  // but the infrastructure is in place for runtime bin tracking.
  covergroup cg @(posedge clk);
    cp_data: coverpoint data;
    // Future: bins low = {[0:3]};
    // Future: bins mid = {[4:11]};
    // Future: bins high = {[12:15]};
  endgroup

  // CHECK: moore.covergroup.inst @cg
  cg cg_inst = new();

  // Test sampling (implicit via clocking event)
  // The runtime will track coverage data when values are sampled.

endmodule

// Test multiple coverpoints in a single covergroup
module test_multi_coverpoint;
  logic clk;
  logic [7:0] addr;
  logic [15:0] data;
  logic wr_en;

  // CHECK: moore.module @test_multi_coverpoint
  // CHECK: moore.covergroup.decl @transaction_cg
  // CHECK:   moore.coverpoint.decl @addr_cp
  // CHECK:   moore.coverpoint.decl @data_cp
  // CHECK:   moore.coverpoint.decl @wr_en_cp

  covergroup transaction_cg @(posedge clk);
    addr_cp: coverpoint addr;
    data_cp: coverpoint data;
    wr_en_cp: coverpoint wr_en;
  endgroup

  // CHECK: moore.covergroup.inst @transaction_cg
  transaction_cg cov = new();

endmodule

// Test covergroup with sampling method
module test_sampling;
  logic clk;
  logic [3:0] value;

  // CHECK: moore.module @test_sampling
  // CHECK: moore.covergroup.decl @sample_cg

  covergroup sample_cg @(posedge clk);
    cp: coverpoint value;
  endgroup

  sample_cg my_cg = new();

  // The coverage infrastructure supports explicit sample() calls
  // which will trigger runtime coverage data collection.

endmodule
