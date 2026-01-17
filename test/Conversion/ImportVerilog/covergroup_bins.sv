// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test covergroup bins conversion to Moore dialect ops.
// IEEE 1800-2017 Section 19.5.1 "Defining bins for values"

module test_covergroup_bins;
  logic [3:0] data;
  logic       clk;

  // Test basic bins specification
  // CHECK: moore.covergroup.decl @cg_bins sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @data_cp : !moore.l4 {
  // CHECK:     moore.coverbin.decl @low kind<bins> values [1, 2, 3]
  // CHECK:     moore.coverbin.decl @mid kind<bins> values [4, 5, 6, 7]
  // CHECK:     moore.coverbin.decl @high kind<bins> values [8, 9, 10, 11]
  // CHECK:   }
  // CHECK: }
  covergroup cg_bins @(posedge clk);
    data_cp: coverpoint data {
      bins low = {1, 2, 3};
      bins mid = {4, 5, 6, 7};
      bins high = {8, 9, 10, 11};
    }
  endgroup

  // Test illegal_bins and ignore_bins
  // CHECK: moore.covergroup.decl @cg_special_bins sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @data_cp : !moore.l4 {
  // CHECK:     moore.coverbin.decl @valid kind<bins> values [1, 2, 3, 4]
  // CHECK:     moore.coverbin.decl @reserved kind<illegal_bins> values [15]
  // CHECK:     moore.coverbin.decl @zero kind<ignore_bins> values [0]
  // CHECK:   }
  // CHECK: }
  covergroup cg_special_bins @(posedge clk);
    data_cp: coverpoint data {
      bins valid = {1, 2, 3, 4};
      illegal_bins reserved = {15};
      ignore_bins zero = {0};
    }
  endgroup

  // Test default bins
  // CHECK: moore.covergroup.decl @cg_default_bins sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @data_cp : !moore.l4 {
  // CHECK:     moore.coverbin.decl @specific kind<bins> values [1, 2]
  // CHECK:     moore.coverbin.decl @others kind<bins> default
  // CHECK:   }
  // CHECK: }
  covergroup cg_default_bins @(posedge clk);
    data_cp: coverpoint data {
      bins specific = {1, 2};
      bins others = default;
    }
  endgroup

endmodule
