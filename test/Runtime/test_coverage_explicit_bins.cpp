//===- test_coverage_explicit_bins.cpp - Test explicit bin coverage -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test for the Moore runtime coverage collection with explicit bins and JSON
// reporting. This tests the enhanced coverage infrastructure that supports
// SystemVerilog-style named bins.
//
//===----------------------------------------------------------------------===//

#include "circt/Runtime/MooreRuntime.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

// Helper macro for assertions with messages
#define CHECK(cond, msg) do { \
  if (!(cond)) { \
    std::cerr << "FAILED: " << (msg) << std::endl; \
    return 1; \
  } \
} while(0)

#define PASS(msg) std::cout << "PASSED: " << msg << std::endl

int main() {
  std::cout << "Testing Moore Runtime Coverage - Explicit Bins\n";
  std::cout << "================================================\n\n";

  // Test 1: Create covergroup with explicit bins
  {
    void *cg = __moore_covergroup_create("explicit_bins_cg", 1);
    CHECK(cg != nullptr, "covergroup create should return non-null");

    // Define explicit bins for a 4-bit data coverpoint
    // Simulating: bins low = {[0:3]}; bins mid = {[4:11]}; bins high = {[12:15]};
    MooreCoverageBin bins[3];
    bins[0].name = "low";
    bins[0].type = MOORE_BIN_RANGE;
    bins[0].low = 0;
    bins[0].high = 3;
    bins[0].hit_count = 0;

    bins[1].name = "mid";
    bins[1].type = MOORE_BIN_RANGE;
    bins[1].low = 4;
    bins[1].high = 11;
    bins[1].hit_count = 0;

    bins[2].name = "high";
    bins[2].type = MOORE_BIN_RANGE;
    bins[2].low = 12;
    bins[2].high = 15;
    bins[2].hit_count = 0;

    __moore_coverpoint_init_with_bins(cg, 0, "cp_data", bins, 3);

    // Initial coverage should be 0%
    double cov = __moore_coverpoint_get_coverage(cg, 0);
    CHECK(cov == 0.0, "initial coverage should be 0%");

    __moore_covergroup_destroy(cg);
    PASS("Coverpoint with explicit bins initialization");
  }

  // Test 2: Sample values and check bin hits
  {
    void *cg = __moore_covergroup_create("bin_hits_cg", 1);
    CHECK(cg != nullptr, "covergroup create should return non-null");

    MooreCoverageBin bins[3];
    bins[0].name = "low";
    bins[0].type = MOORE_BIN_RANGE;
    bins[0].low = 0;
    bins[0].high = 3;
    bins[0].hit_count = 0;

    bins[1].name = "mid";
    bins[1].type = MOORE_BIN_RANGE;
    bins[1].low = 4;
    bins[1].high = 11;
    bins[1].hit_count = 0;

    bins[2].name = "high";
    bins[2].type = MOORE_BIN_RANGE;
    bins[2].low = 12;
    bins[2].high = 15;
    bins[2].hit_count = 0;

    __moore_coverpoint_init_with_bins(cg, 0, "cp_data", bins, 3);

    // Sample values in the "low" range
    __moore_coverpoint_sample(cg, 0, 0);
    __moore_coverpoint_sample(cg, 0, 1);
    __moore_coverpoint_sample(cg, 0, 2);

    // Check bin hits
    int64_t lowHits = __moore_coverpoint_get_bin_hits(cg, 0, 0);
    CHECK(lowHits == 3, "low bin should have 3 hits");

    int64_t midHits = __moore_coverpoint_get_bin_hits(cg, 0, 1);
    CHECK(midHits == 0, "mid bin should have 0 hits");

    int64_t highHits = __moore_coverpoint_get_bin_hits(cg, 0, 2);
    CHECK(highHits == 0, "high bin should have 0 hits");

    // Sample in mid range
    __moore_coverpoint_sample(cg, 0, 5);
    __moore_coverpoint_sample(cg, 0, 8);

    midHits = __moore_coverpoint_get_bin_hits(cg, 0, 1);
    CHECK(midHits == 2, "mid bin should have 2 hits");

    // Coverage should be 2/3 = 66.67%
    double cov = __moore_coverpoint_get_coverage(cg, 0);
    CHECK(std::abs(cov - 66.666666) < 1.0, "coverage should be ~66.67%");

    __moore_covergroup_destroy(cg);
    PASS("Bin hit tracking");
  }

  // Test 3: Value bins (single value)
  {
    void *cg = __moore_covergroup_create("value_bins_cg", 1);
    CHECK(cg != nullptr, "covergroup create should return non-null");

    MooreCoverageBin bins[3];
    bins[0].name = "zero";
    bins[0].type = MOORE_BIN_VALUE;
    bins[0].low = 0;
    bins[0].high = 0;
    bins[0].hit_count = 0;

    bins[1].name = "one";
    bins[1].type = MOORE_BIN_VALUE;
    bins[1].low = 1;
    bins[1].high = 1;
    bins[1].hit_count = 0;

    bins[2].name = "two";
    bins[2].type = MOORE_BIN_VALUE;
    bins[2].low = 2;
    bins[2].high = 2;
    bins[2].hit_count = 0;

    __moore_coverpoint_init_with_bins(cg, 0, "cp_values", bins, 3);

    // Sample exact values
    __moore_coverpoint_sample(cg, 0, 0);
    __moore_coverpoint_sample(cg, 0, 0);
    __moore_coverpoint_sample(cg, 0, 1);
    __moore_coverpoint_sample(cg, 0, 3); // Not in any bin

    CHECK(__moore_coverpoint_get_bin_hits(cg, 0, 0) == 2, "zero bin should have 2 hits");
    CHECK(__moore_coverpoint_get_bin_hits(cg, 0, 1) == 1, "one bin should have 1 hit");
    CHECK(__moore_coverpoint_get_bin_hits(cg, 0, 2) == 0, "two bin should have 0 hits");

    // Coverage should be 2/3 = 66.67%
    double cov = __moore_coverpoint_get_coverage(cg, 0);
    CHECK(std::abs(cov - 66.666666) < 1.0, "coverage should be ~66.67%");

    __moore_covergroup_destroy(cg);
    PASS("Value bin tracking");
  }

  // Test 4: Dynamic bin addition
  {
    void *cg = __moore_covergroup_create("dynamic_bins_cg", 1);
    CHECK(cg != nullptr, "covergroup create should return non-null");

    // Initialize with empty bins
    __moore_coverpoint_init(cg, 0, "cp_dynamic");

    // Add bins dynamically
    __moore_coverpoint_add_bin(cg, 0, "bin_a", MOORE_BIN_RANGE, 0, 10);
    __moore_coverpoint_add_bin(cg, 0, "bin_b", MOORE_BIN_RANGE, 11, 20);

    // Sample values
    __moore_coverpoint_sample(cg, 0, 5);
    __moore_coverpoint_sample(cg, 0, 15);

    CHECK(__moore_coverpoint_get_bin_hits(cg, 0, 0) == 1, "bin_a should have 1 hit");
    CHECK(__moore_coverpoint_get_bin_hits(cg, 0, 1) == 1, "bin_b should have 1 hit");

    __moore_covergroup_destroy(cg);
    PASS("Dynamic bin addition");
  }

  // Test 5: Full coverage (100%)
  {
    void *cg = __moore_covergroup_create("full_cov_cg", 1);
    CHECK(cg != nullptr, "covergroup create should return non-null");

    MooreCoverageBin bins[2];
    bins[0].name = "low";
    bins[0].type = MOORE_BIN_RANGE;
    bins[0].low = 0;
    bins[0].high = 7;
    bins[0].hit_count = 0;

    bins[1].name = "high";
    bins[1].type = MOORE_BIN_RANGE;
    bins[1].low = 8;
    bins[1].high = 15;
    bins[1].hit_count = 0;

    __moore_coverpoint_init_with_bins(cg, 0, "cp_full", bins, 2);

    // Sample values in both ranges
    __moore_coverpoint_sample(cg, 0, 3);  // low
    __moore_coverpoint_sample(cg, 0, 12); // high

    double cov = __moore_coverpoint_get_coverage(cg, 0);
    CHECK(std::abs(cov - 100.0) < 0.01, "coverage should be 100%");

    __moore_covergroup_destroy(cg);
    PASS("Full coverage (100%)");
  }

  // Test 6: JSON report generation (stdout)
  {
    void *cg = __moore_covergroup_create("json_test_cg", 2);
    CHECK(cg != nullptr, "covergroup create should return non-null");

    MooreCoverageBin bins1[2];
    bins1[0].name = "low";
    bins1[0].type = MOORE_BIN_RANGE;
    bins1[0].low = 0;
    bins1[0].high = 127;
    bins1[0].hit_count = 0;

    bins1[1].name = "high";
    bins1[1].type = MOORE_BIN_RANGE;
    bins1[1].low = 128;
    bins1[1].high = 255;
    bins1[1].hit_count = 0;

    __moore_coverpoint_init_with_bins(cg, 0, "data_cp", bins1, 2);
    __moore_coverpoint_init(cg, 1, "addr_cp"); // Auto bins

    // Sample some values
    for (int i = 0; i < 10; ++i) {
      __moore_coverpoint_sample(cg, 0, i * 20);
      __moore_coverpoint_sample(cg, 1, i);
    }

    std::cout << "\n--- JSON Coverage Report ---\n";
    __moore_coverage_report_json_stdout();
    std::cout << "--- End JSON Report ---\n\n";

    __moore_covergroup_destroy(cg);
    PASS("JSON report to stdout");
  }

  // Test 7: JSON string retrieval
  {
    void *cg = __moore_covergroup_create("json_string_cg", 1);
    CHECK(cg != nullptr, "covergroup create should return non-null");

    __moore_coverpoint_init(cg, 0, "test_cp");
    __moore_coverpoint_sample(cg, 0, 42);

    char *json = __moore_coverage_get_json();
    CHECK(json != nullptr, "JSON string should not be null");
    CHECK(std::strstr(json, "coverage_report") != nullptr,
          "JSON should contain 'coverage_report'");
    CHECK(std::strstr(json, "json_string_cg") != nullptr,
          "JSON should contain covergroup name");

    __moore_free(json);
    __moore_covergroup_destroy(cg);
    PASS("JSON string retrieval");
  }

  // Test 8: Multiple covergroups
  {
    void *cg1 = __moore_covergroup_create("multi_cg1", 1);
    void *cg2 = __moore_covergroup_create("multi_cg2", 1);
    CHECK(cg1 != nullptr && cg2 != nullptr, "both covergroups should be created");

    __moore_coverpoint_init(cg1, 0, "cp1");
    __moore_coverpoint_init(cg2, 0, "cp2");

    __moore_coverpoint_sample(cg1, 0, 10);
    __moore_coverpoint_sample(cg2, 0, 20);

    // Both should appear in JSON
    char *json = __moore_coverage_get_json();
    CHECK(std::strstr(json, "multi_cg1") != nullptr, "JSON should contain cg1");
    CHECK(std::strstr(json, "multi_cg2") != nullptr, "JSON should contain cg2");

    __moore_free(json);
    __moore_covergroup_destroy(cg1);
    __moore_covergroup_destroy(cg2);
    PASS("Multiple covergroups");
  }

  // Test 9: Edge cases for bin matching
  {
    void *cg = __moore_covergroup_create("edge_case_cg", 1);
    CHECK(cg != nullptr, "covergroup create should return non-null");

    MooreCoverageBin bins[2];
    bins[0].name = "exact_boundary_low";
    bins[0].type = MOORE_BIN_RANGE;
    bins[0].low = 0;
    bins[0].high = 0;  // Single value as range
    bins[0].hit_count = 0;

    bins[1].name = "exact_boundary_high";
    bins[1].type = MOORE_BIN_RANGE;
    bins[1].low = 100;
    bins[1].high = 100;  // Single value as range
    bins[1].hit_count = 0;

    __moore_coverpoint_init_with_bins(cg, 0, "cp_edge", bins, 2);

    // Test exact boundary values
    __moore_coverpoint_sample(cg, 0, 0);   // Should hit first bin
    __moore_coverpoint_sample(cg, 0, 100); // Should hit second bin
    __moore_coverpoint_sample(cg, 0, 50);  // Should not hit any bin

    CHECK(__moore_coverpoint_get_bin_hits(cg, 0, 0) == 1, "boundary low should have 1 hit");
    CHECK(__moore_coverpoint_get_bin_hits(cg, 0, 1) == 1, "boundary high should have 1 hit");

    __moore_covergroup_destroy(cg);
    PASS("Edge case bin matching");
  }

  // Test 10: Coverage report with bins
  {
    void *cg = __moore_covergroup_create("report_bins_cg", 1);
    CHECK(cg != nullptr, "covergroup create should return non-null");

    MooreCoverageBin bins[3];
    bins[0].name = "low";
    bins[0].type = MOORE_BIN_RANGE;
    bins[0].low = 0;
    bins[0].high = 3;
    bins[0].hit_count = 0;

    bins[1].name = "mid";
    bins[1].type = MOORE_BIN_RANGE;
    bins[1].low = 4;
    bins[1].high = 11;
    bins[1].hit_count = 0;

    bins[2].name = "high";
    bins[2].type = MOORE_BIN_RANGE;
    bins[2].low = 12;
    bins[2].high = 15;
    bins[2].hit_count = 0;

    __moore_coverpoint_init_with_bins(cg, 0, "cp_data", bins, 3);

    // Sample values in all ranges
    __moore_coverpoint_sample(cg, 0, 2);
    __moore_coverpoint_sample(cg, 0, 7);
    __moore_coverpoint_sample(cg, 0, 14);

    std::cout << "\n--- Text Coverage Report ---\n";
    __moore_coverage_report();
    std::cout << "--- End Text Report ---\n\n";

    __moore_covergroup_destroy(cg);
    PASS("Coverage report with bins");
  }

  std::cout << "\n================================================\n";
  std::cout << "All explicit bins tests passed!\n";
  return 0;
}
