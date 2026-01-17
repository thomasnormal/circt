// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Test coverage runtime support infrastructure.
// This tests the SystemVerilog coverage constructs that lower to runtime
// function calls for coverage data collection and reporting.

//===----------------------------------------------------------------------===//
// Basic covergroup with auto bins
//===----------------------------------------------------------------------===//

module BasicCoverage;
  logic clk;
  logic [7:0] data;
  logic [3:0] addr;

  // CHECK: moore.module @BasicCoverage
  // CHECK:   moore.covergroup.inst @basic_cg

  // A basic covergroup that uses automatic bins.
  // Runtime will track: __moore_covergroup_create, __moore_coverpoint_init,
  // and __moore_coverpoint_sample calls.
  covergroup basic_cg @(posedge clk);
    data_cp: coverpoint data;
    addr_cp: coverpoint addr;
  endgroup

  basic_cg cg_inst = new();

  // The runtime maintains:
  // - Per-coverpoint hit counts
  // - Min/max value tracking
  // - Unique value tracking for coverage percentage calculation
endmodule

// CHECK: moore.covergroup.decl @basic_cg
// CHECK:   moore.coverpoint.decl @data_cp
// CHECK:   moore.coverpoint.decl @addr_cp

//===----------------------------------------------------------------------===//
// Coverage with explicit sample method
//===----------------------------------------------------------------------===//

module ExplicitSampling;
  logic clk;
  logic [15:0] transaction_id;
  logic [7:0] payload;
  logic valid;

  // CHECK: moore.module @ExplicitSampling
  // CHECK:   moore.covergroup.inst @transaction_cg

  // Covergroup for transaction monitoring
  covergroup transaction_cg @(posedge clk);
    id_cp: coverpoint transaction_id;
    payload_cp: coverpoint payload;
    valid_cp: coverpoint valid;
  endgroup

  transaction_cg txn_cov = new();

  // Explicit sampling can be triggered programmatically:
  // txn_cov.sample(); // This calls __moore_coverpoint_sample for each coverpoint
  //
  // Coverage can be queried:
  // $display("Coverage: %0.2f%%", txn_cov.get_coverage());
  // This calls __moore_covergroup_get_coverage which returns the average
  // coverage across all coverpoints.
endmodule

// CHECK: moore.covergroup.decl @transaction_cg
// CHECK:   moore.coverpoint.decl @id_cp
// CHECK:   moore.coverpoint.decl @payload_cp
// CHECK:   moore.coverpoint.decl @valid_cp

//===----------------------------------------------------------------------===//
// Multiple covergroup instances
//===----------------------------------------------------------------------===//

module MultiInstance;
  logic clk;
  logic [7:0] data_a, data_b;

  // CHECK: moore.module @MultiInstance

  // Covergroup can be instantiated multiple times
  // Each instance has its own runtime coverage tracking state
  covergroup data_cg @(posedge clk);
    cp: coverpoint data_a;
  endgroup

  // Two instances of the same covergroup type
  data_cg inst_a = new();
  data_cg inst_b = new();

  // Runtime maintains separate coverage data for each instance.
  // __moore_coverage_report() iterates all registered instances.
endmodule

// CHECK: moore.covergroup.decl @data_cg
// CHECK:   moore.coverpoint.decl @cp

//===----------------------------------------------------------------------===//
// Coverage report generation test
//===----------------------------------------------------------------------===//

module CoverageReporting;
  logic clk;
  logic [3:0] cmd;
  logic [7:0] data;
  logic [15:0] addr;

  // CHECK: moore.module @CoverageReporting

  // Comprehensive covergroup for report testing
  covergroup report_cg @(posedge clk);
    cmd_cp: coverpoint cmd;
    data_cp: coverpoint data;
    addr_cp: coverpoint addr;
  endgroup

  report_cg my_cov = new();

  // At end of simulation, coverage can be reported via:
  //
  // 1. Text report to stdout:
  //    __moore_coverage_report()
  //    Output format:
  //    =================================================
  //              Coverage Report
  //    =================================================
  //    Covergroup: report_cg
  //      Overall coverage: 50.00%
  //      Coverpoints: 3
  //        - cmd_cp: 100 hits, 75.00% coverage [range: 0..15, 12 unique values]
  //        - data_cp: 500 hits, 50.00% coverage [range: 0..255, 128 unique values]
  //        - addr_cp: 1000 hits, 25.00% coverage [range: 0..65535, 250 unique values]
  //    =================================================
  //
  // 2. JSON report to file:
  //    __moore_coverage_report_json("coverage.json")
  //    Output includes: covergroups, coverpoints, bins, hit counts, percentages
  //
  // 3. JSON to stdout:
  //    __moore_coverage_report_json_stdout()
  //
  // 4. Get JSON as string:
  //    char *json = __moore_coverage_get_json();
  //    // ... use json ...
  //    __moore_free(json);
endmodule

// CHECK: moore.covergroup.decl @report_cg
// CHECK:   moore.coverpoint.decl @cmd_cp
// CHECK:   moore.coverpoint.decl @data_cp
// CHECK:   moore.coverpoint.decl @addr_cp

//===----------------------------------------------------------------------===//
// Cross coverage placeholder
//===----------------------------------------------------------------------===//

module CrossCoverage;
  logic clk;
  logic [3:0] a, b;

  // CHECK: moore.module @CrossCoverage

  // Cross coverage support is future work.
  // The CoverCrossDeclOp is currently erased during lowering.
  covergroup cross_cg @(posedge clk);
    cp_a: coverpoint a;
    cp_b: coverpoint b;
    // Cross coverage tracks combinations of coverpoint values
    // cross cp_a, cp_b; // Future: generates cross bins
  endgroup

  cross_cg x_cov = new();
endmodule

// CHECK: moore.covergroup.decl @cross_cg
// CHECK:   moore.coverpoint.decl @cp_a
// CHECK:   moore.coverpoint.decl @cp_b

//===----------------------------------------------------------------------===//
// Coverage with narrow and wide types
//===----------------------------------------------------------------------===//

module TypeCoverage;
  logic clk;
  logic single_bit;
  logic [31:0] wide_data;
  logic [63:0] very_wide;

  // CHECK: moore.module @TypeCoverage

  // Test coverage with various bit widths
  covergroup type_cg @(posedge clk);
    bit_cp: coverpoint single_bit;    // 2 possible values
    wide_cp: coverpoint wide_data;    // 2^32 possible values
    vwide_cp: coverpoint very_wide;   // 2^64 possible values
  endgroup

  type_cg t_cov = new();

  // The runtime handles type conversion:
  // - Values are extended/truncated to int64_t for sampling
  // - Coverage percentage is based on unique values seen vs range
  // - For large ranges, coverage is estimated based on observed distribution
endmodule

// CHECK: moore.covergroup.decl @type_cg
// CHECK:   moore.coverpoint.decl @bit_cp
// CHECK:   moore.coverpoint.decl @wide_cp
// CHECK:   moore.coverpoint.decl @vwide_cp
