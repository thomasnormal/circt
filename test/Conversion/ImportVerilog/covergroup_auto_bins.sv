// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test automatic bin creation patterns for coverpoints.
// IEEE 1800-2017 Section 19.5.1 "Defining bins for values"

module test_auto_bins;
  logic [7:0] data;
  logic [3:0] small_data;
  logic       clk;

  // Test array bin syntax: bins x[] = {values}
  // Creates one bin per value in the list
  // CHECK: moore.covergroup.decl @cg_array_bins sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @data_cp : !moore.l8 {
  // CHECK:     moore.coverbin.decl @vals kind<bins> array values [1, 2, 3, 4, 5]
  // CHECK:   }
  // CHECK: }
  covergroup cg_array_bins @(posedge clk);
    data_cp: coverpoint data {
      bins vals[] = {1, 2, 3, 4, 5};
    }
  endgroup

  // Test array bin with range: bins x[] = {[low:high]}
  // Creates one bin per value in the range
  // CHECK: moore.covergroup.decl @cg_array_range sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @data_cp : !moore.l4 {
  // CHECK:     moore.coverbin.decl @range_vals kind<bins> array
  // CHECK:   }
  // CHECK: }
  covergroup cg_array_range @(posedge clk);
    data_cp: coverpoint small_data {
      bins range_vals[] = {[0:7]};
    }
  endgroup

  // Test fixed-count bins: bins x[N] = {values}
  // Creates N bins, each covering a portion of the values
  // CHECK: moore.covergroup.decl @cg_fixed_bins sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @data_cp : !moore.l8 {
  // CHECK:     moore.coverbin.decl @partitioned kind<bins> array num_bins<4>
  // CHECK:   }
  // CHECK: }
  covergroup cg_fixed_bins @(posedge clk);
    data_cp: coverpoint data {
      bins partitioned[4] = {[0:255]};
    }
  endgroup

  // Test auto_bin_max option
  // Limits the number of automatically created bins
  // CHECK: moore.covergroup.decl @cg_auto_bin_max sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @data_cp : !moore.l8 auto_bin_max<16> {
  // CHECK:   }
  // CHECK: }
  covergroup cg_auto_bin_max @(posedge clk);
    data_cp: coverpoint data {
      option.auto_bin_max = 16;
    }
  endgroup

  // Test wildcard bins
  // CHECK: moore.covergroup.decl @cg_wildcard_bins sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @data_cp : !moore.l4 {
  // CHECK:     moore.coverbin.decl @pattern kind<bins> wildcard
  // CHECK:   }
  // CHECK: }
  covergroup cg_wildcard_bins @(posedge clk);
    data_cp: coverpoint small_data {
      wildcard bins pattern = {4'b1???};
    }
  endgroup

  // Test combination of array bins with multiple value sets
  // CHECK: moore.covergroup.decl @cg_multi_array sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @data_cp : !moore.l4 {
  // CHECK:     moore.coverbin.decl @low kind<bins> array values [1, 2, 3]
  // CHECK:     moore.coverbin.decl @high kind<bins> array values [12, 13, 14]
  // CHECK:   }
  // CHECK: }
  covergroup cg_multi_array @(posedge clk);
    data_cp: coverpoint small_data {
      bins low[] = {1, 2, 3};
      bins high[] = {12, 13, 14};
    }
  endgroup

  // Test ignore_bins with array syntax
  // CHECK: moore.covergroup.decl @cg_ignore_array sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @data_cp : !moore.l4 {
  // CHECK:     moore.coverbin.decl @skip kind<ignore_bins> array values [0, 15]
  // CHECK:   }
  // CHECK: }
  covergroup cg_ignore_array @(posedge clk);
    data_cp: coverpoint small_data {
      ignore_bins skip[] = {0, 15};
    }
  endgroup

endmodule
