//===- test_coverage_runtime.cpp - Test coverage runtime ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple test for the Moore runtime coverage collection functions.
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
  std::cout << "Testing Moore Runtime Coverage Collection\n";
  std::cout << "==========================================\n\n";

  // Test 1: Create and destroy covergroup
  {
    void *cg = __moore_covergroup_create("test_cg1", 2);
    CHECK(cg != nullptr, "covergroup create should return non-null");
    __moore_covergroup_destroy(cg);
    PASS("Covergroup create/destroy");
  }

  // Test 2: Create covergroup with zero coverpoints
  {
    void *cg = __moore_covergroup_create("empty_cg", 0);
    CHECK(cg != nullptr, "covergroup with 0 coverpoints should succeed");
    double cov = __moore_covergroup_get_coverage(cg);
    CHECK(cov == 0.0, "empty covergroup should have 0% coverage");
    __moore_covergroup_destroy(cg);
    PASS("Covergroup with zero coverpoints");
  }

  // Test 3: Create covergroup with negative coverpoints (invalid)
  {
    void *cg = __moore_covergroup_create("bad_cg", -1);
    CHECK(cg == nullptr, "covergroup with negative coverpoints should fail");
    PASS("Covergroup with negative coverpoints rejected");
  }

  // Test 4: Initialize coverpoints
  {
    void *cg = __moore_covergroup_create("test_cg2", 2);
    CHECK(cg != nullptr, "covergroup create should succeed");

    __moore_coverpoint_init(cg, 0, "cp0");
    __moore_coverpoint_init(cg, 1, "cp1");

    // Initial coverage should be 0%
    double cov0 = __moore_coverpoint_get_coverage(cg, 0);
    double cov1 = __moore_coverpoint_get_coverage(cg, 1);
    CHECK(cov0 == 0.0, "initial coverage should be 0%");
    CHECK(cov1 == 0.0, "initial coverage should be 0%");

    __moore_covergroup_destroy(cg);
    PASS("Coverpoint initialization");
  }

  // Test 5: Sample values
  {
    void *cg = __moore_covergroup_create("test_cg3", 1);
    CHECK(cg != nullptr, "covergroup create should succeed");

    __moore_coverpoint_init(cg, 0, "cp");

    // Sample some values
    __moore_coverpoint_sample(cg, 0, 10);
    __moore_coverpoint_sample(cg, 0, 20);
    __moore_coverpoint_sample(cg, 0, 15);

    // Coverage should be non-zero
    double cov = __moore_coverpoint_get_coverage(cg, 0);
    CHECK(cov > 0.0, "coverage should be > 0 after sampling");
    CHECK(cov <= 100.0, "coverage should be <= 100%");

    __moore_covergroup_destroy(cg);
    PASS("Value sampling");
  }

  // Test 6: Single value gives 100% coverage
  {
    void *cg = __moore_covergroup_create("test_cg4", 1);
    CHECK(cg != nullptr, "covergroup create should succeed");

    __moore_coverpoint_init(cg, 0, "cp");

    // Sample the same value multiple times
    __moore_coverpoint_sample(cg, 0, 42);
    __moore_coverpoint_sample(cg, 0, 42);
    __moore_coverpoint_sample(cg, 0, 42);

    // Coverage should be 100%
    double cov = __moore_coverpoint_get_coverage(cg, 0);
    CHECK(std::abs(cov - 100.0) < 0.01, "single value should give 100% coverage");

    __moore_covergroup_destroy(cg);
    PASS("Single value 100% coverage");
  }

  // Test 7: Full range coverage
  {
    void *cg = __moore_covergroup_create("test_cg5", 1);
    CHECK(cg != nullptr, "covergroup create should succeed");

    __moore_coverpoint_init(cg, 0, "cp");

    // Sample all values in a range
    for (int i = 0; i <= 10; ++i) {
      __moore_coverpoint_sample(cg, 0, i);
    }

    // Coverage should be 100%
    double cov = __moore_coverpoint_get_coverage(cg, 0);
    CHECK(std::abs(cov - 100.0) < 0.01, "full range should give 100% coverage");

    __moore_covergroup_destroy(cg);
    PASS("Full range coverage");
  }

  // Test 8: Overall covergroup coverage
  {
    void *cg = __moore_covergroup_create("test_cg6", 2);
    CHECK(cg != nullptr, "covergroup create should succeed");

    __moore_coverpoint_init(cg, 0, "cp0");
    __moore_coverpoint_init(cg, 1, "cp1");

    // cp0: single value (100% coverage)
    __moore_coverpoint_sample(cg, 0, 42);

    // cp1: sparse coverage
    __moore_coverpoint_sample(cg, 1, 0);
    __moore_coverpoint_sample(cg, 1, 50);
    __moore_coverpoint_sample(cg, 1, 100);

    // Overall coverage should be average
    double cov = __moore_covergroup_get_coverage(cg);
    CHECK(cov > 0.0, "overall coverage should be > 0");
    CHECK(cov <= 100.0, "overall coverage should be <= 100%");

    __moore_covergroup_destroy(cg);
    PASS("Overall covergroup coverage");
  }

  // Test 9: Null safety
  {
    // These should not crash
    __moore_covergroup_destroy(nullptr);
    __moore_coverpoint_init(nullptr, 0, "bad");
    __moore_coverpoint_sample(nullptr, 0, 100);
    double cov = __moore_coverpoint_get_coverage(nullptr, 0);
    CHECK(cov == 0.0, "null covergroup should return 0% coverage");
    PASS("Null safety");
  }

  // Test 10: Coverage report (just verify it doesn't crash)
  {
    void *cg = __moore_covergroup_create("report_test", 2);
    CHECK(cg != nullptr, "covergroup create should succeed");

    __moore_coverpoint_init(cg, 0, "addr_cp");
    __moore_coverpoint_init(cg, 1, "data_cp");

    // Sample some values
    for (int i = 0; i < 10; ++i) {
      __moore_coverpoint_sample(cg, 0, i * 10);
      __moore_coverpoint_sample(cg, 1, i);
    }

    std::cout << "\n--- Coverage Report Output ---\n";
    __moore_coverage_report();
    std::cout << "--- End Coverage Report ---\n\n";

    __moore_covergroup_destroy(cg);
    PASS("Coverage report generation");
  }

  std::cout << "\n==========================================\n";
  std::cout << "All tests passed!\n";
  return 0;
}
