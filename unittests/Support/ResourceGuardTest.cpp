//===- ResourceGuardTest.cpp - Resource guard tests -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/ResourceGuard.h"
#include "gtest/gtest.h"

using namespace circt;

TEST(ResourceGuardTest, ParseMegabytes) {
  EXPECT_EQ(parseMegabytes("0"), std::optional<uint64_t>(0));
  EXPECT_EQ(parseMegabytes("  123 "), std::optional<uint64_t>(123));
  EXPECT_FALSE(parseMegabytes(""));
  EXPECT_FALSE(parseMegabytes(" "));
  EXPECT_FALSE(parseMegabytes("12MB"));
  EXPECT_FALSE(parseMegabytes("-1"));
}

