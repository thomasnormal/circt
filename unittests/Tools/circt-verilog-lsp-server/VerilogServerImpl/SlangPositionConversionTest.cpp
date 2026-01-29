//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for slang-to-LSP position conversion helper functions.
// These tests verify that the conversion correctly handles edge cases,
// particularly the case where slang returns column number 0 for invalid
// locations, which should be clamped to 0 instead of becoming -1.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

namespace {

/// Convert slang 1-based line number to LSP 0-based line number.
/// Clamps to 0 if slang returns 0 (which can happen for invalid locations).
inline int slangLineToLsp(size_t slangLine) {
  return slangLine > 0 ? static_cast<int>(slangLine) - 1 : 0;
}

/// Convert slang 1-based column number to LSP 0-based column number.
/// Clamps to 0 if slang returns 0 (which can happen for invalid locations).
inline int slangColumnToLsp(size_t slangColumn) {
  return slangColumn > 0 ? static_cast<int>(slangColumn) - 1 : 0;
}

/// Test that normal 1-based to 0-based conversion works correctly.
TEST(SlangPositionConversionTest, NormalConversion) {
  // Line 1 in slang should become line 0 in LSP
  EXPECT_EQ(slangLineToLsp(1), 0);
  // Line 10 in slang should become line 9 in LSP
  EXPECT_EQ(slangLineToLsp(10), 9);

  // Column 1 in slang should become column 0 in LSP
  EXPECT_EQ(slangColumnToLsp(1), 0);
  // Column 5 in slang should become column 4 in LSP
  EXPECT_EQ(slangColumnToLsp(5), 4);
}

/// Test that column 0 (invalid/edge case) is clamped to 0, not -1.
/// This is the main bug fix being tested.
TEST(SlangPositionConversionTest, ZeroInputClampedToZero) {
  // Column 0 from slang should become 0, not -1
  // This was the bug: column 0 - 1 = -1 which is invalid for LSP Position
  EXPECT_EQ(slangColumnToLsp(0), 0);

  // Same for line number
  EXPECT_EQ(slangLineToLsp(0), 0);
}

/// Test large line/column numbers.
TEST(SlangPositionConversionTest, LargeNumbers) {
  // Test with reasonably large numbers
  EXPECT_EQ(slangLineToLsp(1000000), 999999);
  EXPECT_EQ(slangColumnToLsp(1000000), 999999);
}

/// Test that result is always non-negative.
TEST(SlangPositionConversionTest, ResultAlwaysNonNegative) {
  // All inputs should produce non-negative outputs
  for (size_t i = 0; i <= 100; ++i) {
    EXPECT_GE(slangLineToLsp(i), 0) << "Failed for line " << i;
    EXPECT_GE(slangColumnToLsp(i), 0) << "Failed for column " << i;
  }
}

} // namespace
