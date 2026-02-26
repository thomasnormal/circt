// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Test coverage collection with explicit bins.
// This tests the infrastructure for SystemVerilog covergroups with named bins.

module test_coverage;
  logic clk;
  logic [3:0] data;

  // CHECK: moore.module @test_coverage
  // CHECK:   moore.covergroup.inst @cg

  // Basic covergroup with a coverpoint and explicit bins.
  covergroup cg @(posedge clk);
    cp_data: coverpoint data {
      // CHECK: moore.coverbin.decl @low kind<bins> values {{\[\[0, 3\]\]}}
      bins low = {[0:3]};
      // CHECK: moore.coverbin.decl @mid kind<bins> values {{\[\[4, 11\]\]}}
      bins mid = {[4:11]};
      // CHECK: moore.coverbin.decl @high kind<bins> values {{\[\[12, 15\]\]}}
      bins high = {[12:15]};
    }
  endgroup

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
  // CHECK:   moore.covergroup.inst @transaction_cg

  covergroup transaction_cg @(posedge clk);
    addr_cp: coverpoint addr;
    data_cp: coverpoint data;
    wr_en_cp: coverpoint wr_en;
  endgroup

  transaction_cg cov = new();

endmodule

// CHECK: moore.covergroup.decl @transaction_cg
// CHECK:   moore.coverpoint.decl @addr_cp
// CHECK:   moore.coverpoint.decl @data_cp
// CHECK:   moore.coverpoint.decl @wr_en_cp

// Test covergroup with sampling method
module test_sampling;
  logic clk;
  logic [3:0] value;

  // CHECK: moore.module @test_sampling

  covergroup sample_cg @(posedge clk);
    cp: coverpoint value;
  endgroup

  sample_cg my_cg = new();

  // The coverage infrastructure supports explicit sample() calls
  // which will trigger runtime coverage data collection.

endmodule

// CHECK: moore.covergroup.decl @sample_cg
// CHECK:   moore.coverpoint.decl @cp
