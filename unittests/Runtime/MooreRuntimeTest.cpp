//===- MooreRuntimeTest.cpp - Tests for Moore runtime library -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the Moore runtime library functions.
//
//===----------------------------------------------------------------------===//

#include "circt/Runtime/MooreRuntime.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

MooreString makeMooreString(const char *cstr) {
  if (!cstr)
    return {nullptr, 0};
  return {const_cast<char *>(cstr), static_cast<int64_t>(std::strlen(cstr))};
}

//===----------------------------------------------------------------------===//
// String Length Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeStringTest, StringLen) {
  // Test with a regular string
  char data[] = "hello";
  MooreString str = {data, 5};
  EXPECT_EQ(__moore_string_len(&str), 5);

  // Test with empty string
  MooreString empty = {nullptr, 0};
  EXPECT_EQ(__moore_string_len(&empty), 0);

  // Test with null pointer
  EXPECT_EQ(__moore_string_len(nullptr), 0);
}

//===----------------------------------------------------------------------===//
// String Case Conversion Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeStringTest, StringToUpper) {
  char data[] = "Hello World";
  MooreString str = {data, 11};
  MooreString result = __moore_string_toupper(&str);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 11);
  EXPECT_EQ(std::string(result.data, result.len), "HELLO WORLD");

  __moore_free(result.data);
}

//===----------------------------------------------------------------------===//
// DPI Regex Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeDpiRegexTest, RegexExecMatches) {
  MooreString pattern = makeMooreString("test.*pattern");
  void *rexp = uvm_re_comp(&pattern, 0);
  ASSERT_NE(rexp, nullptr);

  MooreString target = makeMooreString("xx test-123-pattern yy");
  int32_t pos = uvm_re_exec(rexp, &target);
  EXPECT_GE(pos, 0);

  MooreString buffer = uvm_re_buffer();
  EXPECT_EQ(std::string(buffer.data, buffer.len), "test-123-pattern");
  __moore_free(buffer.data);

  uvm_re_free(rexp);
}

TEST(MooreRuntimeDpiRegexTest, RegexExecNoMatchClearsBuffer) {
  MooreString pattern = makeMooreString("abc");
  void *rexp = uvm_re_comp(&pattern, 0);
  ASSERT_NE(rexp, nullptr);

  MooreString target = makeMooreString("zzz");
  EXPECT_EQ(uvm_re_exec(rexp, &target), -1);

  MooreString buffer = uvm_re_buffer();
  EXPECT_EQ(buffer.len, 0);
  EXPECT_EQ(buffer.data, nullptr);

  uvm_re_free(rexp);
}

TEST(MooreRuntimeDpiRegexTest, RegexDeglobMatches) {
  MooreString pattern = makeMooreString("foo*bar");
  void *rexp = uvm_re_comp(&pattern, 1);
  ASSERT_NE(rexp, nullptr);

  MooreString target = makeMooreString("fooXYZbar");
  EXPECT_GE(uvm_re_exec(rexp, &target), 0);

  MooreString buffer = uvm_re_buffer();
  EXPECT_EQ(std::string(buffer.data, buffer.len), "fooXYZbar");
  __moore_free(buffer.data);

  uvm_re_free(rexp);
}

TEST(MooreRuntimeDpiRegexTest, RegexInvalidPattern) {
  MooreString pattern = makeMooreString("[unterminated");
  void *rexp = uvm_re_comp(&pattern, 0);
  EXPECT_EQ(rexp, nullptr);
}

TEST(MooreRuntimeDpiRegexTest, RegexCompexecfree) {
  MooreString pattern = makeMooreString("ab.*c");
  MooreString target = makeMooreString("xx abZZc yy");
  int32_t execRet = -2;
  EXPECT_EQ(uvm_re_compexecfree(&pattern, &target, 0, &execRet), 1);
  EXPECT_GE(execRet, 0);
}

TEST(MooreRuntimeDpiRegexTest, RegexDeglobbed) {
  MooreString pattern = makeMooreString("foo*bar?.sv");
  MooreString converted = uvm_re_deglobbed(&pattern, 1);
  EXPECT_EQ(std::string(converted.data, converted.len), "foo.*bar.\\.sv");
  __moore_free(converted.data);
}

//===----------------------------------------------------------------------===//
// DPI Command Line Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeDpiArgsTest, GetNextArgFromEnv) {
  setenv("CIRCT_UVM_ARGS", "arg1 \"two words\" 'three words' arg\\\"4", 1);
  unsetenv("UVM_ARGS");

  int32_t idx = 0;
  MooreString arg1 = uvm_dpi_get_next_arg_c(&idx);
  EXPECT_EQ(std::string(arg1.data, arg1.len), "arg1");
  __moore_free(arg1.data);

  MooreString arg2 = uvm_dpi_get_next_arg_c(&idx);
  EXPECT_EQ(std::string(arg2.data, arg2.len), "two words");
  __moore_free(arg2.data);

  MooreString arg3 = uvm_dpi_get_next_arg_c(&idx);
  EXPECT_EQ(std::string(arg3.data, arg3.len), "three words");
  __moore_free(arg3.data);

  MooreString arg4 = uvm_dpi_get_next_arg_c(&idx);
  EXPECT_EQ(std::string(arg4.data, arg4.len), "arg\"4");
  __moore_free(arg4.data);

  MooreString arg5 = uvm_dpi_get_next_arg_c(&idx);
  EXPECT_EQ(arg5.data, nullptr);
  EXPECT_EQ(arg5.len, 0);
}

TEST(MooreRuntimeDpiArgsTest, GetNextArgFromUvmArgsFallback) {
  unsetenv("CIRCT_UVM_ARGS");
  setenv("UVM_ARGS", "fallback1 fallback2", 1);

  int32_t idx = 0;
  MooreString arg1 = uvm_dpi_get_next_arg_c(&idx);
  EXPECT_EQ(std::string(arg1.data, arg1.len), "fallback1");
  __moore_free(arg1.data);

  MooreString arg2 = uvm_dpi_get_next_arg_c(&idx);
  EXPECT_EQ(std::string(arg2.data, arg2.len), "fallback2");
  __moore_free(arg2.data);

  MooreString arg3 = uvm_dpi_get_next_arg_c(&idx);
  EXPECT_EQ(arg3.data, nullptr);
  EXPECT_EQ(arg3.len, 0);
}

TEST(MooreRuntimeDpiArgsTest, GetNextArgClearsOnEmpty) {
  setenv("CIRCT_UVM_ARGS", "first", 1);
  unsetenv("UVM_ARGS");
  int32_t idx = 0;
  MooreString arg1 = uvm_dpi_get_next_arg_c(&idx);
  EXPECT_EQ(std::string(arg1.data, arg1.len), "first");
  __moore_free(arg1.data);

  setenv("CIRCT_UVM_ARGS", "", 1);
  idx = 0;
  MooreString arg2 = uvm_dpi_get_next_arg_c(&idx);
  EXPECT_EQ(arg2.data, nullptr);
  EXPECT_EQ(arg2.len, 0);
}

//===----------------------------------------------------------------------===//
// VPI Stub Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeVpiTest, VpiStubsReturnDefaults) {
  EXPECT_EQ(vpi_handle_by_name(nullptr, nullptr), nullptr);
  EXPECT_EQ(vpi_handle_by_name("", nullptr), nullptr);
  EXPECT_EQ(vpi_get(0, nullptr), 0);
  EXPECT_EQ(vpi_get_str(0, nullptr), nullptr);
  vpi_value getValue = {0, nullptr};
  EXPECT_EQ(vpi_get_value(nullptr, &getValue), 0);
  vpi_value value = {0, nullptr};
  EXPECT_EQ(vpi_put_value(nullptr, &value, nullptr, 0), 0);
}

TEST(MooreRuntimeVpiTest, VpiHandleName) {
  vpiHandle handle = vpi_handle_by_name("top.sig", nullptr);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(vpi_get(0, handle), 1);
  EXPECT_EQ(std::string(vpi_get_str(0, handle)), "top.sig");
  vpi_release_handle(handle);
}

TEST(MooreRuntimeVpiTest, VpiHandleSeedsHdlMap) {
  vpiHandle handle = vpi_handle_by_name("top.vpi_sig", nullptr);
  ASSERT_NE(handle, nullptr);

  MooreString path = makeMooreString("top.vpi_sig");
  uvm_hdl_data_t value = 99;
  EXPECT_EQ(uvm_hdl_read(&path, &value), 1);
  EXPECT_EQ(value, 0);

  vpi_release_handle(nullptr);
  vpi_release_handle(handle);
}

TEST(MooreRuntimeVpiTest, VpiPutValueUpdatesHdl) {
  vpiHandle handle = vpi_handle_by_name("top.vpi_put", nullptr);
  ASSERT_NE(handle, nullptr);

  uvm_hdl_data_t newValue = 1234;
  vpi_value value = {0, &newValue};
  EXPECT_EQ(vpi_put_value(handle, &value, nullptr, 0), 1);

  MooreString path = makeMooreString("top.vpi_put");
  uvm_hdl_data_t readValue = 0;
  EXPECT_EQ(uvm_hdl_read(&path, &readValue), 1);
  EXPECT_EQ(readValue, newValue);

  vpi_release_handle(handle);
}

TEST(MooreRuntimeVpiTest, VpiPutValueForceFlag) {
  vpiHandle handle = vpi_handle_by_name("top.vpi_force", nullptr);
  ASSERT_NE(handle, nullptr);

  uvm_hdl_data_t newValue = 77;
  vpi_value value = {0, &newValue};
  EXPECT_EQ(vpi_put_value(handle, &value, nullptr, 1), 1);

  MooreString path = makeMooreString("top.vpi_force");
  uvm_hdl_data_t readValue = 0;
  EXPECT_EQ(uvm_hdl_deposit(&path, 12), 1);
  EXPECT_EQ(uvm_hdl_read(&path, &readValue), 1);
  EXPECT_EQ(readValue, newValue);

  vpi_release_handle(handle);
}

TEST(MooreRuntimeVpiTest, VpiPutValueForceRelease) {
  vpiHandle handle = vpi_handle_by_name("top.vpi_force_rel", nullptr);
  ASSERT_NE(handle, nullptr);

  uvm_hdl_data_t newValue = 88;
  vpi_value value = {0, &newValue};
  EXPECT_EQ(vpi_put_value(handle, &value, nullptr, 1), 1);

  MooreString path = makeMooreString("top.vpi_force_rel");
  uvm_hdl_data_t readValue = 0;
  EXPECT_EQ(uvm_hdl_release(&path), 1);
  EXPECT_EQ(uvm_hdl_deposit(&path, 33), 1);
  EXPECT_EQ(uvm_hdl_read(&path, &readValue), 1);
  EXPECT_EQ(readValue, 33);

  vpi_release_handle(handle);
}

TEST(MooreRuntimeVpiTest, VpiGetValueReadsHdl) {
  vpiHandle handle = vpi_handle_by_name("top.vpi_get", nullptr);
  ASSERT_NE(handle, nullptr);

  MooreString path = makeMooreString("top.vpi_get");
  EXPECT_EQ(uvm_hdl_deposit(&path, 55), 1);

  uvm_hdl_data_t readValue = 0;
  vpi_value value = {0, &readValue};
  EXPECT_EQ(vpi_get_value(handle, &value), 1);
  EXPECT_EQ(readValue, 55);

  vpi_release_handle(handle);
}

TEST(MooreRuntimeVpiTest, VpiGetValueNullInput) {
  vpiHandle handle = vpi_handle_by_name("top.vpi_get_null", nullptr);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(vpi_get_value(handle, nullptr), 0);
  vpi_value value = {0, nullptr};
  EXPECT_EQ(vpi_get_value(handle, &value), 0);
  vpi_release_handle(handle);
}

TEST(MooreRuntimeVpiTest, VpiGetStrInvalidHandle) {
  EXPECT_EQ(vpi_get_str(0, nullptr), nullptr);
}

TEST(MooreRuntimeVpiTest, VpiPutValueNullInput) {
  vpiHandle handle = vpi_handle_by_name("top.vpi_null", nullptr);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(vpi_put_value(handle, nullptr, nullptr, 0), 0);
  vpi_value value = {0, nullptr};
  EXPECT_EQ(vpi_put_value(handle, &value, nullptr, 0), 0);
  vpi_release_handle(handle);
}

//===----------------------------------------------------------------------===//
// Randomize RandC Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeRandCTest, RandcCyclesValues) {
  int64_t bitWidth = 2;
  int64_t values[4];
  for (int i = 0; i < 4; ++i) {
    values[i] = __moore_randc_next(values, bitWidth);
  }
  std::sort(std::begin(values), std::end(values));
  EXPECT_EQ(values[0], 0);
  EXPECT_EQ(values[1], 1);
  EXPECT_EQ(values[2], 2);
  EXPECT_EQ(values[3], 3);
}

TEST(MooreRuntimeRandCTest, RandcCyclesRepeat) {
  int64_t bitWidth = 2;
  int64_t values[4];
  for (int i = 0; i < 4; ++i)
    values[i] = __moore_randc_next(values, bitWidth);
  std::sort(std::begin(values), std::end(values));
  EXPECT_EQ(values[0], 0);
  EXPECT_EQ(values[1], 1);
  EXPECT_EQ(values[2], 2);
  EXPECT_EQ(values[3], 3);
}

TEST(MooreRuntimeRandCTest, RandcCyclesFourBit) {
  int64_t bitWidth = 4;
  int64_t values[16];
  for (int i = 0; i < 16; ++i)
    values[i] = __moore_randc_next(values, bitWidth);
  std::sort(std::begin(values), std::end(values));
  for (int i = 0; i < 16; ++i)
    EXPECT_EQ(values[i], i);
}

TEST(MooreRuntimeRandCTest, RandcCyclesFiveBit) {
  int64_t bitWidth = 5;
  int64_t values[32];
  for (int i = 0; i < 32; ++i)
    values[i] = __moore_randc_next(values, bitWidth);
  std::sort(std::begin(values), std::end(values));
  for (int i = 0; i < 32; ++i)
    EXPECT_EQ(values[i], i);
}

TEST(MooreRuntimeRandCTest, RandcCyclesSixBit) {
  int64_t bitWidth = 6;
  int64_t values[64];
  for (int i = 0; i < 64; ++i)
    values[i] = __moore_randc_next(values, bitWidth);
  std::sort(std::begin(values), std::end(values));
  for (int i = 0; i < 64; ++i)
    EXPECT_EQ(values[i], i);
}

TEST(MooreRuntimeRandCTest, RandcWideRangeClamped) {
  int64_t bitWidth = 20;
  int64_t value = __moore_randc_next(&bitWidth, bitWidth);
  uint64_t mask = (1ULL << bitWidth) - 1;
  EXPECT_EQ(static_cast<uint64_t>(value) & ~mask, 0ULL);
}

TEST(MooreRuntimeRandCTest, RandcWideRangeLinearStep) {
  int64_t bitWidth = 20;
  int64_t value1 = __moore_randc_next(&bitWidth, bitWidth);
  int64_t value2 = __moore_randc_next(&bitWidth, bitWidth);
  EXPECT_NE(value1, value2);
}

TEST(MooreRuntimeRandCTest, RandcIndependentFields) {
  int64_t bitWidth = 2;
  int64_t valuesA[4];
  int64_t valuesB[4];
  for (int i = 0; i < 4; ++i) {
    valuesA[i] = __moore_randc_next(valuesA, bitWidth);
    valuesB[i] = __moore_randc_next(valuesB, bitWidth);
  }
  std::sort(std::begin(valuesA), std::end(valuesA));
  std::sort(std::begin(valuesB), std::end(valuesB));
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(valuesA[i], i);
    EXPECT_EQ(valuesB[i], i);
  }
}

TEST(MooreRuntimeRandCTest, RandcBitWidthChangeResetsCycle) {
  int64_t key = 0;
  int64_t bitWidth = 2;
  int64_t values2[4];
  for (int i = 0; i < 4; ++i)
    values2[i] = __moore_randc_next(&key, bitWidth);
  std::sort(std::begin(values2), std::end(values2));
  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(values2[i], i);

  bitWidth = 3;
  int64_t values3[8];
  for (int i = 0; i < 8; ++i)
    values3[i] = __moore_randc_next(&key, bitWidth);
  std::sort(std::begin(values3), std::end(values3));
  for (int i = 0; i < 8; ++i)
    EXPECT_EQ(values3[i], i);
}

//===----------------------------------------------------------------------===//
// DPI HDL Access Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeDpiHdlTest, HdlDepositAndRead) {
  MooreString path = makeMooreString("top.sig");
  EXPECT_EQ(uvm_hdl_check_path(&path), 1);
  EXPECT_EQ(uvm_hdl_deposit(&path, 123), 1);

  uvm_hdl_data_t value = 0;
  EXPECT_EQ(uvm_hdl_read(&path, &value), 1);
  EXPECT_EQ(value, 123);
}

TEST(MooreRuntimeDpiHdlTest, HdlCheckPathCreatesEntry) {
  MooreString path = makeMooreString("top.newsig");
  EXPECT_EQ(uvm_hdl_check_path(&path), 1);

  uvm_hdl_data_t value = 55;
  EXPECT_EQ(uvm_hdl_read(&path, &value), 1);
  EXPECT_EQ(value, 0);
}

TEST(MooreRuntimeDpiHdlTest, HdlCheckPathInvalid) {
  MooreString empty = {nullptr, 0};
  EXPECT_EQ(uvm_hdl_check_path(&empty), 0);
  EXPECT_EQ(uvm_hdl_check_path(nullptr), 0);
}

TEST(MooreRuntimeDpiHdlTest, HdlForceRelease) {
  MooreString path = makeMooreString("top.force_sig");
  EXPECT_EQ(uvm_hdl_force(&path, 77), 1);

  uvm_hdl_data_t value = 0;
  EXPECT_EQ(uvm_hdl_read(&path, &value), 1);
  EXPECT_EQ(value, 77);

  EXPECT_EQ(uvm_hdl_deposit(&path, 33), 1);
  EXPECT_EQ(uvm_hdl_read(&path, &value), 1);
  EXPECT_EQ(value, 77);

  EXPECT_EQ(uvm_hdl_release(&path), 1);
  EXPECT_EQ(uvm_hdl_read(&path, &value), 1);
  EXPECT_EQ(value, 77);
}

TEST(MooreRuntimeDpiHdlTest, HdlReleaseAndReadUnknown) {
  MooreString path = makeMooreString("top.unknown");
  uvm_hdl_data_t value = 99;
  EXPECT_EQ(uvm_hdl_release_and_read(&path, &value), 1);
  EXPECT_EQ(value, 0);
}

TEST(MooreRuntimeDpiHdlTest, HdlReleaseAndReadClearsForce) {
  MooreString path = makeMooreString("top.force_release");
  EXPECT_EQ(uvm_hdl_force(&path, 11), 1);

  uvm_hdl_data_t value = 0;
  EXPECT_EQ(uvm_hdl_release_and_read(&path, &value), 1);
  EXPECT_EQ(value, 11);

  EXPECT_EQ(uvm_hdl_deposit(&path, 22), 1);
  EXPECT_EQ(uvm_hdl_read(&path, &value), 1);
  EXPECT_EQ(value, 22);
}

TEST(MooreRuntimeDpiHdlTest, HdlReadInvalidPath) {
  MooreString empty = {nullptr, 0};
  uvm_hdl_data_t value = 5;
  EXPECT_EQ(uvm_hdl_read(&empty, &value), 0);
  EXPECT_EQ(value, 5);
}

TEST(MooreRuntimeStringTest, StringToLower) {
  char data[] = "Hello World";
  MooreString str = {data, 11};
  MooreString result = __moore_string_tolower(&str);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 11);
  EXPECT_EQ(std::string(result.data, result.len), "hello world");

  __moore_free(result.data);
}

TEST(MooreRuntimeStringTest, EmptyStringCaseConversion) {
  MooreString empty = {nullptr, 0};

  MooreString upper = __moore_string_toupper(&empty);
  EXPECT_EQ(upper.data, nullptr);
  EXPECT_EQ(upper.len, 0);

  MooreString lower = __moore_string_tolower(&empty);
  EXPECT_EQ(lower.data, nullptr);
  EXPECT_EQ(lower.len, 0);
}

//===----------------------------------------------------------------------===//
// String Character Access Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeStringTest, StringGetC) {
  char data[] = "abc";
  MooreString str = {data, 3};

  EXPECT_EQ(__moore_string_getc(&str, 0), 'a');
  EXPECT_EQ(__moore_string_getc(&str, 1), 'b');
  EXPECT_EQ(__moore_string_getc(&str, 2), 'c');

  // Out of bounds
  EXPECT_EQ(__moore_string_getc(&str, 3), 0);
  EXPECT_EQ(__moore_string_getc(&str, -1), 0);

  // Null string
  EXPECT_EQ(__moore_string_getc(nullptr, 0), 0);
}

//===----------------------------------------------------------------------===//
// Substring Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeStringTest, StringSubstr) {
  char data[] = "hello world";
  MooreString str = {data, 11};

  // Normal substring
  MooreString sub = __moore_string_substr(&str, 0, 5);
  ASSERT_NE(sub.data, nullptr);
  EXPECT_EQ(sub.len, 5);
  EXPECT_EQ(std::string(sub.data, sub.len), "hello");
  __moore_free(sub.data);

  // Middle substring
  sub = __moore_string_substr(&str, 6, 5);
  ASSERT_NE(sub.data, nullptr);
  EXPECT_EQ(sub.len, 5);
  EXPECT_EQ(std::string(sub.data, sub.len), "world");
  __moore_free(sub.data);

  // Substring clamped to string bounds
  sub = __moore_string_substr(&str, 8, 10);
  ASSERT_NE(sub.data, nullptr);
  EXPECT_EQ(sub.len, 3); // Only "rld" available
  EXPECT_EQ(std::string(sub.data, sub.len), "rld");
  __moore_free(sub.data);

  // Out of bounds start
  sub = __moore_string_substr(&str, 20, 5);
  EXPECT_EQ(sub.data, nullptr);
  EXPECT_EQ(sub.len, 0);
}

//===----------------------------------------------------------------------===//
// Integer to String Conversion Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeStringTest, StringItoa) {
  MooreString result = __moore_string_itoa(42);
  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(std::string(result.data, result.len), "42");
  __moore_free(result.data);

  result = __moore_string_itoa(-123);
  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(std::string(result.data, result.len), "-123");
  __moore_free(result.data);

  result = __moore_string_itoa(0);
  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(std::string(result.data, result.len), "0");
  __moore_free(result.data);
}

//===----------------------------------------------------------------------===//
// String Concatenation Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeStringTest, StringConcat) {
  char data1[] = "hello";
  char data2[] = " world";
  MooreString lhs = {data1, 5};
  MooreString rhs = {data2, 6};

  MooreString result = __moore_string_concat(&lhs, &rhs);
  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 11);
  EXPECT_EQ(std::string(result.data, result.len), "hello world");
  __moore_free(result.data);
}

TEST(MooreRuntimeStringTest, StringConcatEmpty) {
  char data[] = "hello";
  MooreString str = {data, 5};
  MooreString empty = {nullptr, 0};

  // Concat with empty on right
  MooreString result = __moore_string_concat(&str, &empty);
  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 5);
  EXPECT_EQ(std::string(result.data, result.len), "hello");
  __moore_free(result.data);

  // Concat with empty on left
  result = __moore_string_concat(&empty, &str);
  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 5);
  EXPECT_EQ(std::string(result.data, result.len), "hello");
  __moore_free(result.data);

  // Concat two empties
  result = __moore_string_concat(&empty, &empty);
  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);
}

//===----------------------------------------------------------------------===//
// String Comparison Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeStringTest, StringCmp) {
  char data1[] = "abc";
  char data2[] = "abd";
  char data3[] = "abc";
  char data4[] = "ab";
  MooreString str1 = {data1, 3};
  MooreString str2 = {data2, 3};
  MooreString str3 = {data3, 3};
  MooreString str4 = {data4, 2};

  EXPECT_EQ(__moore_string_cmp(&str1, &str3), 0);  // Equal
  EXPECT_LT(__moore_string_cmp(&str1, &str2), 0);  // abc < abd
  EXPECT_GT(__moore_string_cmp(&str2, &str1), 0);  // abd > abc
  EXPECT_GT(__moore_string_cmp(&str1, &str4), 0);  // abc > ab (longer)
  EXPECT_LT(__moore_string_cmp(&str4, &str1), 0);  // ab < abc (shorter)
}

TEST(MooreRuntimeStringTest, StringCmpEmpty) {
  char data[] = "abc";
  MooreString str = {data, 3};
  MooreString empty = {nullptr, 0};

  EXPECT_EQ(__moore_string_cmp(&empty, &empty), 0);
  EXPECT_LT(__moore_string_cmp(&empty, &str), 0);
  EXPECT_GT(__moore_string_cmp(&str, &empty), 0);
}

//===----------------------------------------------------------------------===//
// String to Integer Conversion Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeStringTest, StringToInt) {
  char data1[] = "42";
  char data2[] = "-123";
  char data3[] = "0";
  char data4[] = "abc";
  MooreString str1 = {data1, 2};
  MooreString str2 = {data2, 4};
  MooreString str3 = {data3, 1};
  MooreString str4 = {data4, 3};

  EXPECT_EQ(__moore_string_to_int(&str1), 42);
  EXPECT_EQ(__moore_string_to_int(&str2), -123);
  EXPECT_EQ(__moore_string_to_int(&str3), 0);
  EXPECT_EQ(__moore_string_to_int(&str4), 0); // Non-numeric

  MooreString empty = {nullptr, 0};
  EXPECT_EQ(__moore_string_to_int(&empty), 0);
  EXPECT_EQ(__moore_string_to_int(nullptr), 0);
}

//===----------------------------------------------------------------------===//
// Queue Sort Tests
//===----------------------------------------------------------------------===//

// Comparison function for sorting int32_t in ascending order
static int compareInt32Asc(const void *a, const void *b) {
  int32_t va = *static_cast<const int32_t *>(a);
  int32_t vb = *static_cast<const int32_t *>(b);
  return (va > vb) - (va < vb);
}

TEST(MooreRuntimeQueueTest, QueueConcat) {
  MooreQueue q1 = {nullptr, 0};
  MooreQueue q2 = {nullptr, 0};
  int64_t elementSize = sizeof(int64_t);

  int64_t a = 10;
  int64_t b = 20;
  int64_t c = 30;
  int64_t d = 40;
  __moore_queue_push_back(&q1, &a, elementSize);
  __moore_queue_push_back(&q1, &b, elementSize);
  __moore_queue_push_back(&q2, &c, elementSize);
  __moore_queue_push_back(&q2, &d, elementSize);

  MooreQueue queues[2] = {q1, q2};
  MooreQueue result = __moore_queue_concat(queues, 2, elementSize);

  ASSERT_EQ(result.len, 4);
  auto *values = static_cast<int64_t *>(result.data);
  EXPECT_EQ(values[0], 10);
  EXPECT_EQ(values[1], 20);
  EXPECT_EQ(values[2], 30);
  EXPECT_EQ(values[3], 40);

  std::free(q1.data);
  std::free(q2.data);
  std::free(result.data);
}

TEST(MooreRuntimeQueueTest, QueueConcatEmptyInputs) {
  MooreQueue result = __moore_queue_concat(nullptr, 0, sizeof(int32_t));
  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);
}

TEST(MooreRuntimeQueueTest, QueueSortIntegers) {
  // Create a queue with unsorted integers
  int32_t data[] = {5, 2, 8, 1, 9, 3};
  MooreQueue queue = {data, 6};

  auto *result = static_cast<MooreQueue *>(
      __moore_queue_sort(&queue, sizeof(int32_t), compareInt32Asc));

  ASSERT_NE(result, nullptr);
  ASSERT_NE(result->data, nullptr);
  EXPECT_EQ(result->len, 6);

  auto *sorted = static_cast<int32_t *>(result->data);
  EXPECT_EQ(sorted[0], 1);
  EXPECT_EQ(sorted[1], 2);
  EXPECT_EQ(sorted[2], 3);
  EXPECT_EQ(sorted[3], 5);
  EXPECT_EQ(sorted[4], 8);
  EXPECT_EQ(sorted[5], 9);

  // Verify original queue is unchanged
  EXPECT_EQ(data[0], 5);
  EXPECT_EQ(data[1], 2);

  __moore_free(result->data);
  __moore_free(result);
}

TEST(MooreRuntimeQueueTest, QueueSortEmpty) {
  // Test with empty queue
  MooreQueue empty = {nullptr, 0};

  auto *result = static_cast<MooreQueue *>(
      __moore_queue_sort(&empty, sizeof(int32_t), compareInt32Asc));

  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->data, nullptr);
  EXPECT_EQ(result->len, 0);

  __moore_free(result);
}

TEST(MooreRuntimeQueueTest, QueueSortAlreadySorted) {
  // Test with already sorted queue
  int32_t data[] = {1, 2, 3, 4, 5};
  MooreQueue queue = {data, 5};

  auto *result = static_cast<MooreQueue *>(
      __moore_queue_sort(&queue, sizeof(int32_t), compareInt32Asc));

  ASSERT_NE(result, nullptr);
  ASSERT_NE(result->data, nullptr);
  EXPECT_EQ(result->len, 5);

  auto *sorted = static_cast<int32_t *>(result->data);
  EXPECT_EQ(sorted[0], 1);
  EXPECT_EQ(sorted[1], 2);
  EXPECT_EQ(sorted[2], 3);
  EXPECT_EQ(sorted[3], 4);
  EXPECT_EQ(sorted[4], 5);

  __moore_free(result->data);
  __moore_free(result);
}

//===----------------------------------------------------------------------===//
// Dynamic Array Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeArrayTest, DynArrayNew) {
  MooreQueue arr = __moore_dyn_array_new(10);
  ASSERT_NE(arr.data, nullptr);
  EXPECT_EQ(arr.len, 10);

  // Verify zeroed
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(static_cast<char *>(arr.data)[i], 0);
  }

  __moore_free(arr.data);
}

TEST(MooreRuntimeArrayTest, DynArrayNewCopy) {
  char source[] = "hello";
  MooreQueue arr = __moore_dyn_array_new_copy(5, source);
  ASSERT_NE(arr.data, nullptr);
  EXPECT_EQ(arr.len, 5);
  EXPECT_EQ(std::string(static_cast<char *>(arr.data), 5), "hello");

  __moore_free(arr.data);
}

TEST(MooreRuntimeArrayTest, DynArrayEmpty) {
  MooreQueue arr = __moore_dyn_array_new(0);
  EXPECT_EQ(arr.data, nullptr);
  EXPECT_EQ(arr.len, 0);

  arr = __moore_dyn_array_new(-1);
  EXPECT_EQ(arr.data, nullptr);
  EXPECT_EQ(arr.len, 0);
}

//===----------------------------------------------------------------------===//
// Random Number Generation Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeRandomTest, Urandom) {
  // Just verify it returns without crashing and produces values
  uint32_t val1 = __moore_urandom();
  uint32_t val2 = __moore_urandom();

  // It's statistically unlikely that two random numbers are the same
  // (but not impossible), so we just check they're valid uint32_t values
  (void)val1;
  (void)val2;
}

TEST(MooreRuntimeRandomTest, UrandomSeeded) {
  // Verify seeded random produces consistent results
  uint32_t val1 = __moore_urandom_seeded(42);
  uint32_t val2 = __moore_urandom_seeded(42);

  // Re-seeding with the same value should produce the same first random number
  EXPECT_EQ(val1, val2);

  // Different seed should produce different value (with high probability)
  uint32_t val3 = __moore_urandom_seeded(123);
  EXPECT_NE(val1, val3);
}

TEST(MooreRuntimeRandomTest, UrandomRange) {
  // Test range [0, 10]
  for (int i = 0; i < 100; ++i) {
    uint32_t val = __moore_urandom_range(10, 0);
    EXPECT_GE(val, 0u);
    EXPECT_LE(val, 10u);
  }

  // Test range [50, 100]
  for (int i = 0; i < 100; ++i) {
    uint32_t val = __moore_urandom_range(100, 50);
    EXPECT_GE(val, 50u);
    EXPECT_LE(val, 100u);
  }

  // Test with min > max (should swap automatically per IEEE 1800-2017)
  for (int i = 0; i < 100; ++i) {
    uint32_t val = __moore_urandom_range(10, 100); // min=100, max=10, swapped
    EXPECT_GE(val, 10u);
    EXPECT_LE(val, 100u);
  }

  // Test single value range
  uint32_t val = __moore_urandom_range(42, 42);
  EXPECT_EQ(val, 42u);
}

TEST(MooreRuntimeRandomTest, Random) {
  // Just verify it returns without crashing
  int32_t val1 = __moore_random();
  int32_t val2 = __moore_random();
  (void)val1;
  (void)val2;
}

TEST(MooreRuntimeRandomTest, RandomSeeded) {
  // Verify seeded random produces consistent results
  int32_t val1 = __moore_random_seeded(42);
  int32_t val2 = __moore_random_seeded(42);

  // Re-seeding with the same value should produce the same first random number
  EXPECT_EQ(val1, val2);
}

//===----------------------------------------------------------------------===//
// Randomization Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeRandomizeTest, RandomizeBasicSuccess) {
  // Test basic randomization of a struct
  struct TestClass {
    int32_t field1;
    int32_t field2;
    int64_t field3;
  };

  TestClass obj = {0, 0, 0};

  // Randomize should return 1 for success
  int32_t result = __moore_randomize_basic(&obj, sizeof(TestClass));
  EXPECT_EQ(result, 1);

  // It's statistically unlikely all fields remain zero after randomization
  // (but not impossible - we just check the function ran without crashing)
}

TEST(MooreRuntimeRandomizeTest, RandomizeBasicNullPointer) {
  // Null pointer should return 0 (failure)
  int32_t result = __moore_randomize_basic(nullptr, 100);
  EXPECT_EQ(result, 0);
}

TEST(MooreRuntimeRandomizeTest, RandomizeBasicZeroSize) {
  // Zero size should return 0 (failure)
  int32_t value = 42;
  int32_t result = __moore_randomize_basic(&value, 0);
  EXPECT_EQ(result, 0);

  // Value should remain unchanged
  EXPECT_EQ(value, 42);
}

TEST(MooreRuntimeRandomizeTest, RandomizeBasicNegativeSize) {
  // Negative size should return 0 (failure)
  int32_t value = 42;
  int32_t result = __moore_randomize_basic(&value, -1);
  EXPECT_EQ(result, 0);

  // Value should remain unchanged
  EXPECT_EQ(value, 42);
}

TEST(MooreRuntimeRandomizeTest, RandomizeBasicModifiesMemory) {
  // Test that randomization actually modifies memory
  // We use a large buffer to make it statistically unlikely all bytes stay zero
  uint8_t buffer[256];
  std::memset(buffer, 0, sizeof(buffer));

  int32_t result = __moore_randomize_basic(buffer, sizeof(buffer));
  EXPECT_EQ(result, 1);

  // Count non-zero bytes - should have some
  int nonZeroCount = 0;
  for (size_t i = 0; i < sizeof(buffer); ++i) {
    if (buffer[i] != 0)
      ++nonZeroCount;
  }

  // With 256 random bytes, probability of all being zero is (1/256)^256
  // which is essentially impossible. We check for at least some non-zero.
  EXPECT_GT(nonZeroCount, 0);
}

TEST(MooreRuntimeRandomizeTest, RandomizeBasicOddSize) {
  // Test randomization with sizes not divisible by 4 (tests remainder handling)
  uint8_t buffer1[1] = {0};
  uint8_t buffer2[3] = {0, 0, 0};
  uint8_t buffer5[5] = {0, 0, 0, 0, 0};

  EXPECT_EQ(__moore_randomize_basic(buffer1, 1), 1);
  EXPECT_EQ(__moore_randomize_basic(buffer2, 3), 1);
  EXPECT_EQ(__moore_randomize_basic(buffer5, 5), 1);

  // Functions completed without crash - success
}

TEST(MooreRuntimeRandomizeTest, RandomizeBasicSeededConsistency) {
  // Test that seeding the RNG produces consistent randomization results
  struct TestClass {
    int32_t a;
    int32_t b;
    int32_t c;
    int32_t d;
  };

  TestClass obj1, obj2;

  // Seed, then randomize
  __moore_urandom_seeded(12345);
  __moore_randomize_basic(&obj1, sizeof(TestClass));

  // Re-seed with same value, then randomize again
  __moore_urandom_seeded(12345);
  __moore_randomize_basic(&obj2, sizeof(TestClass));

  // Both objects should have identical random values
  EXPECT_EQ(obj1.a, obj2.a);
  EXPECT_EQ(obj1.b, obj2.b);
  EXPECT_EQ(obj1.c, obj2.c);
  EXPECT_EQ(obj1.d, obj2.d);
}

//===----------------------------------------------------------------------===//
// Dynamic Cast / RTTI Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeRTTITest, DynCastCheckSameType) {
  // Cast to the same type should always succeed
  EXPECT_TRUE(__moore_dyn_cast_check(1, 1, 0));
  EXPECT_TRUE(__moore_dyn_cast_check(5, 5, 2));
}

TEST(MooreRuntimeRTTITest, DynCastCheckDerived) {
  // Derived type (higher ID) casting to base type (lower ID) should succeed
  // Type IDs are assigned topologically: base classes get lower IDs
  EXPECT_TRUE(__moore_dyn_cast_check(2, 1, 0)); // derived(2) to base(1)
  EXPECT_TRUE(__moore_dyn_cast_check(5, 1, 0)); // deeper derived to base
}

TEST(MooreRuntimeRTTITest, DynCastCheckBaseToDeriveFails) {
  // Base type (lower ID) casting to derived type (higher ID) should fail
  // You can't upcast - a base object is not an instance of its derived class
  EXPECT_FALSE(__moore_dyn_cast_check(1, 2, 0)); // base(1) to derived(2)
  EXPECT_FALSE(__moore_dyn_cast_check(1, 5, 0)); // base to deeper derived
}

TEST(MooreRuntimeRTTITest, DynCastCheckNullTypeIds) {
  // Null type IDs (0) should always fail
  EXPECT_FALSE(__moore_dyn_cast_check(0, 1, 0)); // null src
  EXPECT_FALSE(__moore_dyn_cast_check(1, 0, 0)); // null target
  EXPECT_FALSE(__moore_dyn_cast_check(0, 0, 0)); // both null
}

TEST(MooreRuntimeRTTITest, DynCastCheckSiblingTypes) {
  // Sibling types (same base but different derived) should fail
  // In a class hierarchy like: Base(1) -> A(2), Base(1) -> B(3)
  // Casting A to B should fail because A is not B
  // Note: With simple ID comparison (>=), this would incorrectly succeed
  // if A(2) is cast to B(3) where B > A. But casting A(2) to B(3) correctly
  // fails because 2 < 3.
  // This tests that src < target fails (can't downcast to sibling)
  EXPECT_FALSE(__moore_dyn_cast_check(2, 3, 1)); // A(2) to B(3) - fail
}

//===----------------------------------------------------------------------===//
// Array Locator Method Tests
//===----------------------------------------------------------------------===//

// Predicate that returns true for values greater than a threshold
static bool greaterThanPredicate(void *element, void *userData) {
  int32_t value = *static_cast<int32_t *>(element);
  int32_t threshold = *static_cast<int32_t *>(userData);
  return value > threshold;
}

// Predicate that returns true for even values
static bool isEvenPredicate(void *element, void * /*userData*/) {
  int32_t value = *static_cast<int32_t *>(element);
  return (value % 2) == 0;
}

TEST(MooreRuntimeArrayLocatorTest, FindAllWithPredicate) {
  int32_t data[] = {1, 5, 3, 8, 2, 9, 4};
  MooreQueue queue = {data, 7};

  int32_t threshold = 4;
  MooreQueue result = __moore_array_locator(&queue, sizeof(int32_t),
                                            greaterThanPredicate, &threshold,
                                            0, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 3); // 5, 8, 9 are > 4

  auto *values = static_cast<int32_t *>(result.data);
  EXPECT_EQ(values[0], 5);
  EXPECT_EQ(values[1], 8);
  EXPECT_EQ(values[2], 9);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindFirstWithPredicate) {
  int32_t data[] = {1, 5, 3, 8, 2, 9, 4};
  MooreQueue queue = {data, 7};

  int32_t threshold = 4;
  MooreQueue result = __moore_array_locator(&queue, sizeof(int32_t),
                                            greaterThanPredicate, &threshold,
                                            1, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 1);

  auto *values = static_cast<int32_t *>(result.data);
  EXPECT_EQ(values[0], 5); // First value > 4

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindLastWithPredicate) {
  int32_t data[] = {1, 5, 3, 8, 2, 9, 4};
  MooreQueue queue = {data, 7};

  int32_t threshold = 4;
  MooreQueue result = __moore_array_locator(&queue, sizeof(int32_t),
                                            greaterThanPredicate, &threshold,
                                            2, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 1);

  auto *values = static_cast<int32_t *>(result.data);
  EXPECT_EQ(values[0], 9); // Last value > 4

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindIndices) {
  int32_t data[] = {2, 5, 4, 8, 1, 6};
  MooreQueue queue = {data, 6};

  MooreQueue result = __moore_array_locator(&queue, sizeof(int32_t),
                                            isEvenPredicate, nullptr,
                                            0, true);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 4); // indices 0, 2, 3, 5 are even

  auto *indices = static_cast<int64_t *>(result.data);
  EXPECT_EQ(indices[0], 0);
  EXPECT_EQ(indices[1], 2);
  EXPECT_EQ(indices[2], 3);
  EXPECT_EQ(indices[3], 5);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindFirstIndex) {
  int32_t data[] = {1, 3, 4, 8, 5, 6};
  MooreQueue queue = {data, 6};

  MooreQueue result = __moore_array_locator(&queue, sizeof(int32_t),
                                            isEvenPredicate, nullptr,
                                            1, true);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 1);

  auto *indices = static_cast<int64_t *>(result.data);
  EXPECT_EQ(indices[0], 2); // First even at index 2

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindLastIndex) {
  int32_t data[] = {2, 3, 4, 8, 5, 6};
  MooreQueue queue = {data, 6};

  MooreQueue result = __moore_array_locator(&queue, sizeof(int32_t),
                                            isEvenPredicate, nullptr,
                                            2, true);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 1);

  auto *indices = static_cast<int64_t *>(result.data);
  EXPECT_EQ(indices[0], 5); // Last even at index 5

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindEqAll) {
  int32_t data[] = {1, 5, 3, 5, 2, 5, 4};
  MooreQueue queue = {data, 7};

  int32_t value = 5;
  MooreQueue result = __moore_array_find_eq(&queue, sizeof(int32_t),
                                            &value, 0, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 3); // Three 5s

  auto *values = static_cast<int32_t *>(result.data);
  EXPECT_EQ(values[0], 5);
  EXPECT_EQ(values[1], 5);
  EXPECT_EQ(values[2], 5);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindEqFirst) {
  int32_t data[] = {1, 5, 3, 5, 2, 5, 4};
  MooreQueue queue = {data, 7};

  int32_t value = 5;
  MooreQueue result = __moore_array_find_eq(&queue, sizeof(int32_t),
                                            &value, 1, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 1);
  EXPECT_EQ(*static_cast<int32_t *>(result.data), 5);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindEqLast) {
  int32_t data[] = {1, 5, 3, 5, 2, 5, 4};
  MooreQueue queue = {data, 7};

  int32_t value = 5;
  MooreQueue result = __moore_array_find_eq(&queue, sizeof(int32_t),
                                            &value, 2, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 1);
  EXPECT_EQ(*static_cast<int32_t *>(result.data), 5);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindEqIndices) {
  int32_t data[] = {1, 5, 3, 5, 2, 5, 4};
  MooreQueue queue = {data, 7};

  int32_t value = 5;
  MooreQueue result = __moore_array_find_eq(&queue, sizeof(int32_t),
                                            &value, 0, true);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 3); // Indices 1, 3, 5

  auto *indices = static_cast<int64_t *>(result.data);
  EXPECT_EQ(indices[0], 1);
  EXPECT_EQ(indices[1], 3);
  EXPECT_EQ(indices[2], 5);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindEqNotFound) {
  int32_t data[] = {1, 2, 3, 4, 5};
  MooreQueue queue = {data, 5};

  int32_t value = 10;
  MooreQueue result = __moore_array_find_eq(&queue, sizeof(int32_t),
                                            &value, 0, false);

  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);
}

TEST(MooreRuntimeArrayLocatorTest, ArrayMinUnsigned) {
  int32_t data[] = {5, 2, 8, 1, 9, 3};
  MooreQueue queue = {data, 6};

  MooreQueue result = __moore_array_min(&queue, sizeof(int32_t), false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 1);
  EXPECT_EQ(*static_cast<int32_t *>(result.data), 1);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, ArrayMaxUnsigned) {
  int32_t data[] = {5, 2, 8, 1, 9, 3};
  MooreQueue queue = {data, 6};

  MooreQueue result = __moore_array_max(&queue, sizeof(int32_t), false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 1);
  EXPECT_EQ(*static_cast<int32_t *>(result.data), 9);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, ArrayMinSigned) {
  int32_t data[] = {5, -2, 8, -10, 9, 3};
  MooreQueue queue = {data, 6};

  MooreQueue result = __moore_array_min(&queue, sizeof(int32_t), true);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 1);
  EXPECT_EQ(*static_cast<int32_t *>(result.data), -10);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, ArrayMaxSigned) {
  int32_t data[] = {5, -2, 8, -10, 9, 3};
  MooreQueue queue = {data, 6};

  MooreQueue result = __moore_array_max(&queue, sizeof(int32_t), true);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 1);
  EXPECT_EQ(*static_cast<int32_t *>(result.data), 9);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, ArrayUnique) {
  int32_t data[] = {1, 2, 3, 2, 1, 4, 3, 5};
  MooreQueue queue = {data, 8};

  MooreQueue result = __moore_array_unique(&queue, sizeof(int32_t));

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 5); // 1, 2, 3, 4, 5

  auto *values = static_cast<int32_t *>(result.data);
  EXPECT_EQ(values[0], 1);
  EXPECT_EQ(values[1], 2);
  EXPECT_EQ(values[2], 3);
  EXPECT_EQ(values[3], 4);
  EXPECT_EQ(values[4], 5);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, ArrayUniqueIndex) {
  int32_t data[] = {1, 2, 3, 2, 1, 4, 3, 5};
  MooreQueue queue = {data, 8};

  MooreQueue result = __moore_array_unique_index(&queue, sizeof(int32_t));

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 5); // Indices 0, 1, 2, 5, 7

  auto *indices = static_cast<int64_t *>(result.data);
  EXPECT_EQ(indices[0], 0); // First occurrence of 1
  EXPECT_EQ(indices[1], 1); // First occurrence of 2
  EXPECT_EQ(indices[2], 2); // First occurrence of 3
  EXPECT_EQ(indices[3], 5); // First occurrence of 4
  EXPECT_EQ(indices[4], 7); // First occurrence of 5

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, EmptyArrayHandling) {
  MooreQueue empty = {nullptr, 0};

  // All locator functions should handle empty arrays gracefully
  MooreQueue result = __moore_array_locator(&empty, sizeof(int32_t),
                                            isEvenPredicate, nullptr,
                                            0, false);
  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);

  int32_t value = 5;
  result = __moore_array_find_eq(&empty, sizeof(int32_t), &value, 0, false);
  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);

  result = __moore_array_min(&empty, sizeof(int32_t), false);
  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);

  result = __moore_array_max(&empty, sizeof(int32_t), false);
  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);

  result = __moore_array_unique(&empty, sizeof(int32_t));
  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);

  result = __moore_array_unique_index(&empty, sizeof(int32_t));
  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);
}

//===----------------------------------------------------------------------===//
// Array Find with Comparison Predicate Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeArrayLocatorTest, FindCmpEqual) {
  int32_t data[] = {1, 5, 3, 5, 2, 5, 4};
  MooreQueue queue = {data, 7};

  int32_t value = 5;
  // cmpMode 0 = equal
  MooreQueue result = __moore_array_find_cmp(&queue, sizeof(int32_t),
                                             &value, MOORE_CMP_EQ, 0, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 3); // Three 5s

  auto *values = static_cast<int32_t *>(result.data);
  EXPECT_EQ(values[0], 5);
  EXPECT_EQ(values[1], 5);
  EXPECT_EQ(values[2], 5);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindCmpNotEqual) {
  int32_t data[] = {1, 5, 3, 5, 2, 5, 4};
  MooreQueue queue = {data, 7};

  int32_t value = 5;
  // cmpMode 1 = not equal
  MooreQueue result = __moore_array_find_cmp(&queue, sizeof(int32_t),
                                             &value, MOORE_CMP_NE, 0, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 4); // 1, 3, 2, 4 are != 5

  auto *values = static_cast<int32_t *>(result.data);
  EXPECT_EQ(values[0], 1);
  EXPECT_EQ(values[1], 3);
  EXPECT_EQ(values[2], 2);
  EXPECT_EQ(values[3], 4);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindCmpSignedGreaterThan) {
  int32_t data[] = {-5, 2, 8, -1, 9, 3};
  MooreQueue queue = {data, 6};

  int32_t value = 2;
  // cmpMode 2 = signed greater than
  MooreQueue result = __moore_array_find_cmp(&queue, sizeof(int32_t),
                                             &value, MOORE_CMP_SGT, 0, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 3); // 8, 9, 3 are > 2

  auto *values = static_cast<int32_t *>(result.data);
  EXPECT_EQ(values[0], 8);
  EXPECT_EQ(values[1], 9);
  EXPECT_EQ(values[2], 3);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindCmpSignedGreaterOrEqual) {
  int32_t data[] = {-5, 2, 8, -1, 9, 3};
  MooreQueue queue = {data, 6};

  int32_t value = 2;
  // cmpMode 3 = signed greater than or equal
  MooreQueue result = __moore_array_find_cmp(&queue, sizeof(int32_t),
                                             &value, MOORE_CMP_SGE, 0, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 4); // 2, 8, 9, 3 are >= 2

  auto *values = static_cast<int32_t *>(result.data);
  EXPECT_EQ(values[0], 2);
  EXPECT_EQ(values[1], 8);
  EXPECT_EQ(values[2], 9);
  EXPECT_EQ(values[3], 3);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindCmpSignedLessThan) {
  int32_t data[] = {-5, 2, 8, -1, 9, 3};
  MooreQueue queue = {data, 6};

  int32_t value = 2;
  // cmpMode 4 = signed less than
  MooreQueue result = __moore_array_find_cmp(&queue, sizeof(int32_t),
                                             &value, MOORE_CMP_SLT, 0, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 2); // -5, -1 are < 2

  auto *values = static_cast<int32_t *>(result.data);
  EXPECT_EQ(values[0], -5);
  EXPECT_EQ(values[1], -1);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindCmpSignedLessOrEqual) {
  int32_t data[] = {-5, 2, 8, -1, 9, 3};
  MooreQueue queue = {data, 6};

  int32_t value = 2;
  // cmpMode 5 = signed less than or equal
  MooreQueue result = __moore_array_find_cmp(&queue, sizeof(int32_t),
                                             &value, MOORE_CMP_SLE, 0, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 3); // -5, 2, -1 are <= 2

  auto *values = static_cast<int32_t *>(result.data);
  EXPECT_EQ(values[0], -5);
  EXPECT_EQ(values[1], 2);
  EXPECT_EQ(values[2], -1);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindCmpFirstMode) {
  int32_t data[] = {1, 5, 3, 8, 2, 9, 4};
  MooreQueue queue = {data, 7};

  int32_t value = 4;
  // locatorMode 1 = first
  MooreQueue result = __moore_array_find_cmp(&queue, sizeof(int32_t),
                                             &value, MOORE_CMP_SGT, 1, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 1); // First > 4 is 5

  EXPECT_EQ(*static_cast<int32_t *>(result.data), 5);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindCmpLastMode) {
  int32_t data[] = {1, 5, 3, 8, 2, 9, 4};
  MooreQueue queue = {data, 7};

  int32_t value = 4;
  // locatorMode 2 = last
  MooreQueue result = __moore_array_find_cmp(&queue, sizeof(int32_t),
                                             &value, MOORE_CMP_SGT, 2, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 1); // Last > 4 is 9

  EXPECT_EQ(*static_cast<int32_t *>(result.data), 9);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindCmpReturnIndices) {
  int32_t data[] = {1, 5, 3, 8, 2, 9, 4};
  MooreQueue queue = {data, 7};

  int32_t value = 4;
  // returnIndices = true
  MooreQueue result = __moore_array_find_cmp(&queue, sizeof(int32_t),
                                             &value, MOORE_CMP_SGT, 0, true);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 3); // Indices of 5, 8, 9 which are 1, 3, 5

  auto *indices = static_cast<int64_t *>(result.data);
  EXPECT_EQ(indices[0], 1); // Index of 5
  EXPECT_EQ(indices[1], 3); // Index of 8
  EXPECT_EQ(indices[2], 5); // Index of 9

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindCmpEmpty) {
  MooreQueue empty = {nullptr, 0};

  int32_t value = 5;
  MooreQueue result = __moore_array_find_cmp(&empty, sizeof(int32_t),
                                             &value, MOORE_CMP_EQ, 0, false);

  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);
}

TEST(MooreRuntimeArrayLocatorTest, FindCmpNotFound) {
  int32_t data[] = {1, 2, 3, 4, 5};
  MooreQueue queue = {data, 5};

  int32_t value = 10;
  // No elements > 10
  MooreQueue result = __moore_array_find_cmp(&queue, sizeof(int32_t),
                                             &value, MOORE_CMP_SGT, 0, false);

  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);
}

//===----------------------------------------------------------------------===//
// Array Field Locator Tests (field-based predicates)
//===----------------------------------------------------------------------===//

// Simple struct to simulate a class object with a field
struct TestItem {
  int32_t x;
  int32_t y;
};

TEST(MooreRuntimeArrayLocatorTest, FindFieldCmpEqual) {
  // Simulate an array of class handles (pointers to objects)
  TestItem item1 = {10, 100};
  TestItem item2 = {20, 200};
  TestItem item3 = {10, 300};
  TestItem item4 = {30, 400};

  TestItem *items[] = {&item1, &item2, &item3, &item4};
  MooreQueue queue = {items, 4};

  // Find items where item.x == 10
  int32_t searchValue = 10;
  int64_t fieldOffset = offsetof(TestItem, x);

  MooreQueue result = __moore_array_find_field_cmp(
      &queue, sizeof(TestItem *), fieldOffset, sizeof(int32_t), &searchValue,
      MOORE_CMP_EQ, 0, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 2);

  auto **found = static_cast<TestItem **>(result.data);
  EXPECT_EQ(found[0]->x, 10);
  EXPECT_EQ(found[1]->x, 10);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindFieldCmpFirstEqual) {
  TestItem item1 = {10, 100};
  TestItem item2 = {20, 200};
  TestItem item3 = {10, 300};
  TestItem item4 = {30, 400};

  TestItem *items[] = {&item1, &item2, &item3, &item4};
  MooreQueue queue = {items, 4};

  // Find first item where item.x == 10
  int32_t searchValue = 10;
  int64_t fieldOffset = offsetof(TestItem, x);

  MooreQueue result = __moore_array_find_field_cmp(
      &queue, sizeof(TestItem *), fieldOffset, sizeof(int32_t), &searchValue,
      MOORE_CMP_EQ, 1, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 1);

  auto **found = static_cast<TestItem **>(result.data);
  EXPECT_EQ(found[0]->y, 100); // First item with x==10

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindFieldCmpLastEqual) {
  TestItem item1 = {10, 100};
  TestItem item2 = {20, 200};
  TestItem item3 = {10, 300};
  TestItem item4 = {30, 400};

  TestItem *items[] = {&item1, &item2, &item3, &item4};
  MooreQueue queue = {items, 4};

  // Find last item where item.x == 10
  int32_t searchValue = 10;
  int64_t fieldOffset = offsetof(TestItem, x);

  MooreQueue result = __moore_array_find_field_cmp(
      &queue, sizeof(TestItem *), fieldOffset, sizeof(int32_t), &searchValue,
      MOORE_CMP_EQ, 2, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 1);

  auto **found = static_cast<TestItem **>(result.data);
  EXPECT_EQ(found[0]->y, 300); // Last item with x==10

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindFieldCmpGreater) {
  TestItem item1 = {10, 100};
  TestItem item2 = {25, 200};
  TestItem item3 = {15, 300};
  TestItem item4 = {30, 400};

  TestItem *items[] = {&item1, &item2, &item3, &item4};
  MooreQueue queue = {items, 4};

  // Find items where item.x > 20
  int32_t threshold = 20;
  int64_t fieldOffset = offsetof(TestItem, x);

  MooreQueue result = __moore_array_find_field_cmp(
      &queue, sizeof(TestItem *), fieldOffset, sizeof(int32_t), &threshold,
      MOORE_CMP_SGT, 0, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 2); // items with x=25 and x=30

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindFieldCmpWithSecondField) {
  TestItem item1 = {10, 100};
  TestItem item2 = {20, 250};
  TestItem item3 = {10, 300};
  TestItem item4 = {30, 150};

  TestItem *items[] = {&item1, &item2, &item3, &item4};
  MooreQueue queue = {items, 4};

  // Find items where item.y >= 200
  int32_t threshold = 200;
  int64_t fieldOffset = offsetof(TestItem, y);

  MooreQueue result = __moore_array_find_field_cmp(
      &queue, sizeof(TestItem *), fieldOffset, sizeof(int32_t), &threshold,
      MOORE_CMP_SGE, 0, false);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 2); // items with y=250 and y=300

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindFieldCmpIndices) {
  TestItem item1 = {10, 100};
  TestItem item2 = {20, 200};
  TestItem item3 = {10, 300};
  TestItem item4 = {30, 400};

  TestItem *items[] = {&item1, &item2, &item3, &item4};
  MooreQueue queue = {items, 4};

  // Find indices where item.x == 10
  int32_t searchValue = 10;
  int64_t fieldOffset = offsetof(TestItem, x);

  MooreQueue result = __moore_array_find_field_cmp(
      &queue, sizeof(TestItem *), fieldOffset, sizeof(int32_t), &searchValue,
      MOORE_CMP_EQ, 0, true);

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 2);

  auto *indices = static_cast<int64_t *>(result.data);
  EXPECT_EQ(indices[0], 0);
  EXPECT_EQ(indices[1], 2);

  __moore_free(result.data);
}

TEST(MooreRuntimeArrayLocatorTest, FindFieldCmpNotFound) {
  TestItem item1 = {10, 100};
  TestItem item2 = {20, 200};

  TestItem *items[] = {&item1, &item2};
  MooreQueue queue = {items, 2};

  // Find items where item.x == 99 (none exist)
  int32_t searchValue = 99;
  int64_t fieldOffset = offsetof(TestItem, x);

  MooreQueue result = __moore_array_find_field_cmp(
      &queue, sizeof(TestItem *), fieldOffset, sizeof(int32_t), &searchValue,
      MOORE_CMP_EQ, 0, false);

  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);
}

//===----------------------------------------------------------------------===//
// Constraint Solving Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeConstraintTest, ConstraintCheckRangeValid) {
  // Value within range
  EXPECT_EQ(__moore_constraint_check_range(5, 0, 10), 1);
  EXPECT_EQ(__moore_constraint_check_range(0, 0, 10), 1);   // At min
  EXPECT_EQ(__moore_constraint_check_range(10, 0, 10), 1);  // At max
  EXPECT_EQ(__moore_constraint_check_range(5, 5, 5), 1);    // Single value range
}

TEST(MooreRuntimeConstraintTest, ConstraintCheckRangeInvalid) {
  // Value outside range
  EXPECT_EQ(__moore_constraint_check_range(-1, 0, 10), 0);  // Below min
  EXPECT_EQ(__moore_constraint_check_range(11, 0, 10), 0);  // Above max
  EXPECT_EQ(__moore_constraint_check_range(100, 0, 10), 0); // Way above
}

TEST(MooreRuntimeConstraintTest, ConstraintCheckRangeNegative) {
  // Negative ranges
  EXPECT_EQ(__moore_constraint_check_range(-5, -10, 0), 1);
  EXPECT_EQ(__moore_constraint_check_range(-10, -10, -5), 1);
  EXPECT_EQ(__moore_constraint_check_range(-5, -10, -5), 1);
  EXPECT_EQ(__moore_constraint_check_range(-11, -10, -5), 0);
  EXPECT_EQ(__moore_constraint_check_range(-4, -10, -5), 0);
}

TEST(MooreRuntimeConstraintTest, RandomizeWithRangeBasic) {
  // Test that randomized values are within range
  for (int i = 0; i < 100; ++i) {
    int64_t val = __moore_randomize_with_range(10, 20);
    EXPECT_GE(val, 10);
    EXPECT_LE(val, 20);
  }
}

TEST(MooreRuntimeConstraintTest, RandomizeWithRangeNegative) {
  // Test with negative range
  for (int i = 0; i < 100; ++i) {
    int64_t val = __moore_randomize_with_range(-100, -50);
    EXPECT_GE(val, -100);
    EXPECT_LE(val, -50);
  }
}

TEST(MooreRuntimeConstraintTest, RandomizeWithRangeMixed) {
  // Test with range spanning zero
  for (int i = 0; i < 100; ++i) {
    int64_t val = __moore_randomize_with_range(-10, 10);
    EXPECT_GE(val, -10);
    EXPECT_LE(val, 10);
  }
}

TEST(MooreRuntimeConstraintTest, RandomizeWithRangeSingleValue) {
  // When min >= max, should return min
  EXPECT_EQ(__moore_randomize_with_range(42, 42), 42);
  EXPECT_EQ(__moore_randomize_with_range(50, 40), 50); // min > max
}

TEST(MooreRuntimeConstraintTest, RandomizeWithRangeSeeded) {
  // Test reproducibility with seeding
  __moore_urandom_seeded(12345);
  int64_t val1 = __moore_randomize_with_range(0, 1000);

  __moore_urandom_seeded(12345);
  int64_t val2 = __moore_randomize_with_range(0, 1000);

  EXPECT_EQ(val1, val2);
}

TEST(MooreRuntimeConstraintTest, RandomizeWithModuloBasic) {
  // Test that result satisfies the modulo constraint
  for (int i = 0; i < 100; ++i) {
    int64_t val = __moore_randomize_with_modulo(7, 3);
    EXPECT_EQ(val % 7, 3);
  }
}

TEST(MooreRuntimeConstraintTest, RandomizeWithModuloZeroRemainder) {
  // Test with zero remainder (value % mod == 0)
  for (int i = 0; i < 100; ++i) {
    int64_t val = __moore_randomize_with_modulo(5, 0);
    EXPECT_EQ(val % 5, 0);
  }
}

TEST(MooreRuntimeConstraintTest, RandomizeWithModuloLargeModulo) {
  // Test with larger modulo values
  for (int i = 0; i < 100; ++i) {
    int64_t val = __moore_randomize_with_modulo(1000, 123);
    EXPECT_EQ(val % 1000, 123);
  }
}

TEST(MooreRuntimeConstraintTest, RandomizeWithModuloInvalidMod) {
  // Invalid modulo (mod <= 0) should return remainder
  EXPECT_EQ(__moore_randomize_with_modulo(0, 5), 5);
  EXPECT_EQ(__moore_randomize_with_modulo(-1, 5), 5);
}

TEST(MooreRuntimeConstraintTest, RandomizeWithModuloSeeded) {
  // Test reproducibility with seeding
  __moore_urandom_seeded(54321);
  int64_t val1 = __moore_randomize_with_modulo(13, 7);

  __moore_urandom_seeded(54321);
  int64_t val2 = __moore_randomize_with_modulo(13, 7);

  EXPECT_EQ(val1, val2);
  EXPECT_EQ(val1 % 13, 7);
}

TEST(MooreRuntimeConstraintTest, RandomizeWithModuloNegativeRemainder) {
  // Test with negative remainder (should be normalized)
  for (int i = 0; i < 100; ++i) {
    int64_t val = __moore_randomize_with_modulo(10, -3);
    // -3 mod 10 = 7 (normalized)
    EXPECT_EQ(val % 10, 7);
  }
}

//===----------------------------------------------------------------------===//
// Constraint Solving with Iteration Limits Tests
//===----------------------------------------------------------------------===//

// Predicate that accepts values divisible by a given number
static bool predicateDivisibleBy(int64_t value, void *userData) {
  int64_t divisor = *static_cast<int64_t *>(userData);
  return (value % divisor) == 0;
}

// Predicate that always returns false (unsatisfiable)
static bool predicateNever(int64_t /*value*/, void * /*userData*/) {
  return false;
}

// Predicate that always returns true (always satisfiable)
static bool predicateAlways(int64_t /*value*/, void * /*userData*/) {
  return true;
}

// Predicate that accepts only a specific value
static bool predicateEquals(int64_t value, void *userData) {
  int64_t target = *static_cast<int64_t *>(userData);
  return value == target;
}

TEST(MooreRuntimeConstraintIterationTest, GetSetIterationLimit) {
  // Reset to default
  __moore_constraint_set_iteration_limit(0);
  EXPECT_EQ(__moore_constraint_get_iteration_limit(),
            MOORE_CONSTRAINT_DEFAULT_ITERATION_LIMIT);

  // Set custom limit
  __moore_constraint_set_iteration_limit(500);
  EXPECT_EQ(__moore_constraint_get_iteration_limit(), 500);

  // Set negative (should reset to default)
  __moore_constraint_set_iteration_limit(-1);
  EXPECT_EQ(__moore_constraint_get_iteration_limit(),
            MOORE_CONSTRAINT_DEFAULT_ITERATION_LIMIT);

  // Reset for other tests
  __moore_constraint_set_iteration_limit(0);
}

TEST(MooreRuntimeConstraintIterationTest, GetResetStats) {
  __moore_constraint_reset_stats();
  MooreConstraintStats *stats = __moore_constraint_get_stats();
  ASSERT_NE(stats, nullptr);

  EXPECT_EQ(stats->totalAttempts, 0);
  EXPECT_EQ(stats->successfulSolves, 0);
  EXPECT_EQ(stats->fallbackCount, 0);
  EXPECT_EQ(stats->iterationLimitHits, 0);
}

TEST(MooreRuntimeConstraintIterationTest, WarningsEnabledFlag) {
  // Should be enabled by default
  EXPECT_TRUE(__moore_constraint_warnings_enabled());

  // Disable warnings
  __moore_constraint_set_warnings_enabled(false);
  EXPECT_FALSE(__moore_constraint_warnings_enabled());

  // Re-enable warnings
  __moore_constraint_set_warnings_enabled(true);
  EXPECT_TRUE(__moore_constraint_warnings_enabled());
}

TEST(MooreRuntimeConstraintIterationTest, RandomizeWithConstraintNoPredicate) {
  __moore_constraint_reset_stats();

  // Without predicate, should just return value in range
  int32_t result = -1;
  int64_t value =
      __moore_randomize_with_constraint(10, 20, nullptr, nullptr, 0, &result);

  EXPECT_GE(value, 10);
  EXPECT_LE(value, 20);
  EXPECT_EQ(result, MOORE_CONSTRAINT_SUCCESS);

  MooreConstraintStats *stats = __moore_constraint_get_stats();
  EXPECT_EQ(stats->totalAttempts, 1);
  EXPECT_EQ(stats->successfulSolves, 1);
  EXPECT_EQ(stats->fallbackCount, 0);
}

TEST(MooreRuntimeConstraintIterationTest, RandomizeWithConstraintSatisfiable) {
  __moore_constraint_reset_stats();

  // Constraint: value must be divisible by 5
  int64_t divisor = 5;
  int32_t result = -1;

  for (int i = 0; i < 100; ++i) {
    int64_t value = __moore_randomize_with_constraint(
        0, 100, predicateDivisibleBy, &divisor, 0, &result);

    EXPECT_GE(value, 0);
    EXPECT_LE(value, 100);
    EXPECT_EQ(value % 5, 0) << "Value should be divisible by 5";
    EXPECT_EQ(result, MOORE_CONSTRAINT_SUCCESS);
  }

  MooreConstraintStats *stats = __moore_constraint_get_stats();
  EXPECT_EQ(stats->totalAttempts, 100);
  EXPECT_EQ(stats->successfulSolves, 100);
  EXPECT_EQ(stats->iterationLimitHits, 0);
}

TEST(MooreRuntimeConstraintIterationTest, RandomizeWithConstraintUnsatisfiable) {
  __moore_constraint_reset_stats();
  // Disable warnings for this test
  __moore_constraint_set_warnings_enabled(false);

  // Set a small iteration limit for faster test
  __moore_constraint_set_iteration_limit(100);

  int32_t result = -1;
  // Unsatisfiable constraint (predicate always returns false)
  int64_t value = __moore_randomize_with_constraint(0, 100, predicateNever,
                                                     nullptr, 0, &result);

  // Should return fallback value (still in range)
  EXPECT_GE(value, 0);
  EXPECT_LE(value, 100);
  EXPECT_EQ(result, MOORE_CONSTRAINT_ITERATION_LIMIT);

  MooreConstraintStats *stats = __moore_constraint_get_stats();
  EXPECT_EQ(stats->totalAttempts, 1);
  EXPECT_EQ(stats->iterationLimitHits, 1);
  EXPECT_EQ(stats->fallbackCount, 1);

  // Re-enable warnings and reset limit
  __moore_constraint_set_warnings_enabled(true);
  __moore_constraint_set_iteration_limit(0);
}

TEST(MooreRuntimeConstraintIterationTest, RandomizeWithConstraintAlwaysSatisfied) {
  __moore_constraint_reset_stats();

  int32_t result = -1;
  // Always satisfied constraint should succeed quickly
  int64_t value = __moore_randomize_with_constraint(0, 100, predicateAlways,
                                                     nullptr, 0, &result);

  EXPECT_GE(value, 0);
  EXPECT_LE(value, 100);
  EXPECT_EQ(result, MOORE_CONSTRAINT_SUCCESS);

  MooreConstraintStats *stats = __moore_constraint_get_stats();
  EXPECT_EQ(stats->successfulSolves, 1);
  EXPECT_EQ(stats->lastIterations, 1); // Should succeed on first try
}

TEST(MooreRuntimeConstraintIterationTest, RandomizeWithConstraintSpecificValue) {
  __moore_constraint_reset_stats();
  __moore_urandom_seeded(42); // Seed for reproducibility

  // Looking for a specific value - may take several iterations
  int64_t target = 50;
  int32_t result = -1;
  int64_t value = __moore_randomize_with_constraint(0, 100, predicateEquals,
                                                     &target, 0, &result);

  // With a reasonable iteration limit, should eventually find target
  EXPECT_EQ(value, target);
  EXPECT_EQ(result, MOORE_CONSTRAINT_SUCCESS);

  MooreConstraintStats *stats = __moore_constraint_get_stats();
  EXPECT_GT(stats->lastIterations, 0);
}

TEST(MooreRuntimeConstraintIterationTest, RandomizeWithConstraintCustomLimit) {
  __moore_constraint_reset_stats();
  __moore_constraint_set_warnings_enabled(false);

  // Use custom iteration limit (small)
  int32_t result = -1;
  int64_t value = __moore_randomize_with_constraint(
      0, 100, predicateNever, nullptr, 5, // Only 5 iterations
      &result);

  EXPECT_GE(value, 0);
  EXPECT_LE(value, 100);
  EXPECT_EQ(result, MOORE_CONSTRAINT_ITERATION_LIMIT);

  MooreConstraintStats *stats = __moore_constraint_get_stats();
  EXPECT_EQ(stats->lastIterations, 5);

  __moore_constraint_set_warnings_enabled(true);
}

TEST(MooreRuntimeConstraintIterationTest, RandomizeWithConstraintSwappedRange) {
  __moore_constraint_reset_stats();

  // Test with min > max (should be swapped internally)
  int32_t result = -1;
  int64_t value =
      __moore_randomize_with_constraint(100, 50, nullptr, nullptr, 0, &result);

  EXPECT_GE(value, 50);
  EXPECT_LE(value, 100);
  EXPECT_EQ(result, MOORE_CONSTRAINT_SUCCESS);
}

TEST(MooreRuntimeConstraintIterationTest, RandomizeWithRangesConstrained) {
  __moore_constraint_reset_stats();

  // Multiple ranges: [0-10], [50-60], [90-100]
  int64_t ranges[] = {0, 10, 50, 60, 90, 100};
  int64_t divisor = 5;
  int32_t result = -1;

  for (int i = 0; i < 50; ++i) {
    int64_t value = __moore_randomize_with_ranges_constrained(
        ranges, 3, predicateDivisibleBy, &divisor, 0, &result);

    // Value should be in one of the ranges
    bool inRange = (value >= 0 && value <= 10) ||
                   (value >= 50 && value <= 60) ||
                   (value >= 90 && value <= 100);
    EXPECT_TRUE(inRange) << "Value " << value << " not in any range";

    // Value should satisfy constraint
    EXPECT_EQ(value % 5, 0);
    EXPECT_EQ(result, MOORE_CONSTRAINT_SUCCESS);
  }
}

TEST(MooreRuntimeConstraintIterationTest, RandomizeWithRangesConstrainedNullRanges) {
  __moore_constraint_reset_stats();

  int32_t result = -1;
  int64_t value = __moore_randomize_with_ranges_constrained(
      nullptr, 3, predicateAlways, nullptr, 0, &result);

  EXPECT_EQ(value, 0); // Should return 0 for invalid input
  EXPECT_EQ(result, MOORE_CONSTRAINT_FALLBACK);

  MooreConstraintStats *stats = __moore_constraint_get_stats();
  EXPECT_EQ(stats->fallbackCount, 1);
}

TEST(MooreRuntimeConstraintIterationTest, RandomizeWithRangesConstrainedZeroRanges) {
  __moore_constraint_reset_stats();

  int64_t ranges[] = {0, 10};
  int32_t result = -1;
  int64_t value = __moore_randomize_with_ranges_constrained(
      ranges, 0, predicateAlways, nullptr, 0, &result);

  EXPECT_EQ(value, 0);
  EXPECT_EQ(result, MOORE_CONSTRAINT_FALLBACK);
}

TEST(MooreRuntimeConstraintIterationTest, RandomizeWithRangesConstrainedNoPredicate) {
  __moore_constraint_reset_stats();

  int64_t ranges[] = {10, 20, 30, 40};
  int32_t result = -1;
  int64_t value = __moore_randomize_with_ranges_constrained(
      ranges, 2, nullptr, nullptr, 0, &result);

  bool inRange = (value >= 10 && value <= 20) || (value >= 30 && value <= 40);
  EXPECT_TRUE(inRange);
  EXPECT_EQ(result, MOORE_CONSTRAINT_SUCCESS);
}

TEST(MooreRuntimeConstraintIterationTest, RandomizeWithRangesConstrainedUnsatisfiable) {
  __moore_constraint_reset_stats();
  __moore_constraint_set_warnings_enabled(false);
  __moore_constraint_set_iteration_limit(50);

  int64_t ranges[] = {0, 10, 20, 30};
  int32_t result = -1;
  int64_t value = __moore_randomize_with_ranges_constrained(
      ranges, 2, predicateNever, nullptr, 0, &result);

  // Should still return value in range (fallback)
  bool inRange = (value >= 0 && value <= 10) || (value >= 20 && value <= 30);
  EXPECT_TRUE(inRange);
  EXPECT_EQ(result, MOORE_CONSTRAINT_ITERATION_LIMIT);

  MooreConstraintStats *stats = __moore_constraint_get_stats();
  EXPECT_EQ(stats->iterationLimitHits, 1);

  __moore_constraint_set_warnings_enabled(true);
  __moore_constraint_set_iteration_limit(0);
}

TEST(MooreRuntimeConstraintIterationTest, StatsAccumulateAcrossCalls) {
  __moore_constraint_reset_stats();
  __moore_constraint_set_warnings_enabled(false);
  __moore_constraint_set_iteration_limit(10);

  // Make several successful calls
  for (int i = 0; i < 5; ++i) {
    __moore_randomize_with_constraint(0, 100, predicateAlways, nullptr, 0,
                                       nullptr);
  }

  // Make several failing calls
  for (int i = 0; i < 3; ++i) {
    __moore_randomize_with_constraint(0, 100, predicateNever, nullptr, 0,
                                       nullptr);
  }

  MooreConstraintStats *stats = __moore_constraint_get_stats();
  EXPECT_EQ(stats->totalAttempts, 8);
  EXPECT_EQ(stats->successfulSolves, 5);
  EXPECT_EQ(stats->iterationLimitHits, 3);
  EXPECT_EQ(stats->fallbackCount, 3);

  __moore_constraint_set_warnings_enabled(true);
  __moore_constraint_set_iteration_limit(0);
}

TEST(MooreRuntimeConstraintIterationTest, ResultOutCanBeNull) {
  __moore_constraint_reset_stats();

  // Passing nullptr for resultOut should not crash
  int64_t value =
      __moore_randomize_with_constraint(0, 100, predicateAlways, nullptr, 0,
                                         nullptr);
  EXPECT_GE(value, 0);
  EXPECT_LE(value, 100);

  // Same for ranges version
  int64_t ranges[] = {0, 10};
  value = __moore_randomize_with_ranges_constrained(ranges, 1, predicateAlways,
                                                     nullptr, 0, nullptr);
  EXPECT_GE(value, 0);
  EXPECT_LE(value, 10);
}

//===----------------------------------------------------------------------===//
// Coverage Collection Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCoverageTest, CovergroupCreateDestroy) {
  // Test basic create and destroy
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  // Destroy should not crash
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CovergroupCreateZeroCoverpoints) {
  // Test with zero coverpoints
  void *cg = __moore_covergroup_create("empty_cg", 0);
  ASSERT_NE(cg, nullptr);

  // Should have 0% coverage with no coverpoints
  double cov = __moore_covergroup_get_coverage(cg);
  EXPECT_DOUBLE_EQ(cov, 0.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CovergroupCreateNegative) {
  // Test with negative coverpoints (invalid)
  void *cg = __moore_covergroup_create("bad_cg", -1);
  EXPECT_EQ(cg, nullptr);
}

TEST(MooreRuntimeCoverageTest, CoverpointInit) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  // Initialize coverpoints
  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  // Verify we can get coverage (should be 0% initially)
  double cov0 = __moore_coverpoint_get_coverage(cg, 0);
  double cov1 = __moore_coverpoint_get_coverage(cg, 1);
  EXPECT_DOUBLE_EQ(cov0, 0.0);
  EXPECT_DOUBLE_EQ(cov1, 0.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CoverpointInitInvalidIndex) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  // These should not crash (invalid indices are ignored)
  __moore_coverpoint_init(cg, -1, "bad");
  __moore_coverpoint_init(cg, 5, "bad");
  __moore_coverpoint_init(nullptr, 0, "bad");

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CoverpointSample) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp");

  // Sample some values
  __moore_coverpoint_sample(cg, 0, 10);
  __moore_coverpoint_sample(cg, 0, 20);
  __moore_coverpoint_sample(cg, 0, 15);

  // Coverage should be non-zero after sampling
  // With values 10, 15, 20, range is 10-20 (11 values), 3 unique values
  // Coverage = 3/11 * 100 = 27.27%
  double cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_GT(cov, 0.0);
  EXPECT_LE(cov, 100.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CoverpointSampleSingleValue) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp");

  // Sample the same value multiple times
  __moore_coverpoint_sample(cg, 0, 42);
  __moore_coverpoint_sample(cg, 0, 42);
  __moore_coverpoint_sample(cg, 0, 42);

  // Coverage should be 100% (single value = full coverage of range)
  double cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov, 100.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CoverpointSampleFullRange) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp");

  // Sample all values in a small range
  for (int i = 0; i <= 10; ++i) {
    __moore_coverpoint_sample(cg, 0, i);
  }

  // Coverage should be 100% (all values in range covered)
  double cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov, 100.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CoverpointSampleInvalidIndex) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp");

  // These should not crash (invalid indices are ignored)
  __moore_coverpoint_sample(cg, -1, 100);
  __moore_coverpoint_sample(cg, 5, 100);
  __moore_coverpoint_sample(nullptr, 0, 100);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CovergroupGetCoverage) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  // Sample different amounts for each coverpoint
  // cp0: single value (100% coverage)
  __moore_coverpoint_sample(cg, 0, 42);

  // cp1: sparse coverage (3 values in range 0-100)
  __moore_coverpoint_sample(cg, 1, 0);
  __moore_coverpoint_sample(cg, 1, 50);
  __moore_coverpoint_sample(cg, 1, 100);

  // Overall coverage should be average of individual coverpoints
  double cov = __moore_covergroup_get_coverage(cg);
  EXPECT_GT(cov, 0.0);
  EXPECT_LE(cov, 100.0);

  // cp0 = 100%, cp1 = 3/101 * 100 = ~2.97%, average = ~51.49%
  // Just verify it's in a reasonable range
  EXPECT_GT(cov, 40.0);
  EXPECT_LT(cov, 60.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CovergroupDestroyNull) {
  // Should not crash when destroying null
  __moore_covergroup_destroy(nullptr);
}

TEST(MooreRuntimeCoverageTest, CoverageReportNoGroups) {
  // Should not crash when no covergroups are registered
  // Note: This test assumes registeredCovergroups is empty after
  // destroying all previously created covergroups.
  // Just verify it doesn't crash - output goes to stdout
  __moore_coverage_report();
}

TEST(MooreRuntimeCoverageTest, MultipleCovergroups) {
  // Create multiple covergroups
  void *cg1 = __moore_covergroup_create("cg1", 1);
  void *cg2 = __moore_covergroup_create("cg2", 2);
  ASSERT_NE(cg1, nullptr);
  ASSERT_NE(cg2, nullptr);

  __moore_coverpoint_init(cg1, 0, "cp1_0");
  __moore_coverpoint_init(cg2, 0, "cp2_0");
  __moore_coverpoint_init(cg2, 1, "cp2_1");

  // Sample values
  __moore_coverpoint_sample(cg1, 0, 1);
  __moore_coverpoint_sample(cg2, 0, 2);
  __moore_coverpoint_sample(cg2, 1, 3);

  // Both covergroups should have valid coverage
  double cov1 = __moore_covergroup_get_coverage(cg1);
  double cov2 = __moore_covergroup_get_coverage(cg2);
  EXPECT_GT(cov1, 0.0);
  EXPECT_GT(cov2, 0.0);

  // Cleanup
  __moore_covergroup_destroy(cg1);
  __moore_covergroup_destroy(cg2);
}

TEST(MooreRuntimeCoverageTest, NegativeValuesSample) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp");

  // Sample negative values
  __moore_coverpoint_sample(cg, 0, -10);
  __moore_coverpoint_sample(cg, 0, -5);
  __moore_coverpoint_sample(cg, 0, 0);
  __moore_coverpoint_sample(cg, 0, 5);
  __moore_coverpoint_sample(cg, 0, 10);

  // Coverage should be calculable for negative ranges
  // Range is -10 to 10 (21 values), 5 unique values
  // Coverage = 5/21 * 100 = ~23.8%
  double cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_GT(cov, 0.0);
  EXPECT_LT(cov, 30.0);

  __moore_covergroup_destroy(cg);
}

//===----------------------------------------------------------------------===//
// Cross Coverage Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCrossCoverageTest, CrossCreate) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  int32_t cpIndices[] = {0, 1};
  int32_t crossIdx = __moore_cross_create(cg, "cross01", cpIndices, 2);
  EXPECT_GE(crossIdx, 0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCrossCoverageTest, CrossCreateInvalidIndices) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  // Invalid coverpoint index
  int32_t badIndices[] = {0, 5};
  int32_t crossIdx = __moore_cross_create(cg, "bad_cross", badIndices, 2);
  EXPECT_EQ(crossIdx, -1);

  // Null indices
  crossIdx = __moore_cross_create(cg, "null_cross", nullptr, 2);
  EXPECT_EQ(crossIdx, -1);

  // Less than 2 coverpoints
  int32_t singleIdx[] = {0};
  crossIdx = __moore_cross_create(cg, "single_cross", singleIdx, 1);
  EXPECT_EQ(crossIdx, -1);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCrossCoverageTest, CrossSample) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  int32_t cpIndices[] = {0, 1};
  int32_t crossIdx = __moore_cross_create(cg, "cross01", cpIndices, 2);
  EXPECT_GE(crossIdx, 0);

  // Sample values for both coverpoints
  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 1, 10);

  // Sample the cross with values [1, 10]
  int64_t cpValues[] = {1, 10};
  __moore_cross_sample(cg, cpValues, 2);

  // Should have 1 cross bin hit
  int64_t binsHit = __moore_cross_get_bins_hit(cg, crossIdx);
  EXPECT_EQ(binsHit, 1);

  // Sample a different combination
  __moore_coverpoint_sample(cg, 0, 2);
  __moore_coverpoint_sample(cg, 1, 20);
  int64_t cpValues2[] = {2, 20};
  __moore_cross_sample(cg, cpValues2, 2);

  // Should have 2 cross bins hit now
  binsHit = __moore_cross_get_bins_hit(cg, crossIdx);
  EXPECT_EQ(binsHit, 2);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCrossCoverageTest, CrossGetCoverage) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  int32_t cpIndices[] = {0, 1};
  int32_t crossIdx = __moore_cross_create(cg, "cross01", cpIndices, 2);
  EXPECT_GE(crossIdx, 0);

  // Sample 2 values for cp0 and 2 values for cp1 (4 cross combinations possible)
  __moore_coverpoint_sample(cg, 0, 0);
  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 1, 10);
  __moore_coverpoint_sample(cg, 1, 11);

  // Sample 2 out of 4 combinations
  int64_t vals1[] = {0, 10};
  int64_t vals2[] = {1, 11};
  __moore_cross_sample(cg, vals1, 2);
  __moore_cross_sample(cg, vals2, 2);

  // Coverage should be 50% (2 out of 4 possible combinations)
  double coverage = __moore_cross_get_coverage(cg, crossIdx);
  EXPECT_DOUBLE_EQ(coverage, 50.0);

  __moore_covergroup_destroy(cg);
}

//===----------------------------------------------------------------------===//
// Coverage Reset Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCoverageTest, CoverpointReset) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp");

  // Sample some values
  __moore_coverpoint_sample(cg, 0, 10);
  __moore_coverpoint_sample(cg, 0, 20);
  __moore_coverpoint_sample(cg, 0, 30);

  // Verify we have coverage
  double covBefore = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_GT(covBefore, 0.0);

  // Reset the coverpoint
  __moore_coverpoint_reset(cg, 0);

  // Coverage should be 0 after reset
  double covAfter = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(covAfter, 0.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CovergroupReset) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  // Sample values
  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 1, 2);

  // Verify coverage before reset
  double cov0Before = __moore_coverpoint_get_coverage(cg, 0);
  double cov1Before = __moore_coverpoint_get_coverage(cg, 1);
  EXPECT_GT(cov0Before, 0.0);
  EXPECT_GT(cov1Before, 0.0);

  // Reset the entire covergroup
  __moore_covergroup_reset(cg);

  // All coverpoints should have 0 coverage
  double cov0After = __moore_coverpoint_get_coverage(cg, 0);
  double cov1After = __moore_coverpoint_get_coverage(cg, 1);
  EXPECT_DOUBLE_EQ(cov0After, 0.0);
  EXPECT_DOUBLE_EQ(cov1After, 0.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CoverpointResetInvalidIndex) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp");

  // These should not crash
  __moore_coverpoint_reset(cg, -1);
  __moore_coverpoint_reset(cg, 5);
  __moore_coverpoint_reset(nullptr, 0);

  __moore_covergroup_destroy(cg);
}

//===----------------------------------------------------------------------===//
// Coverage Goal Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCoverageTest, CovergroupGoalDefault) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  // Default goal should be 100%
  double goal = __moore_covergroup_get_goal(cg);
  EXPECT_DOUBLE_EQ(goal, 100.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CovergroupSetGoal) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  // Set custom goal
  __moore_covergroup_set_goal(cg, 80.0);
  double goal = __moore_covergroup_get_goal(cg);
  EXPECT_DOUBLE_EQ(goal, 80.0);

  // Goal should be clamped to [0, 100]
  __moore_covergroup_set_goal(cg, -10.0);
  EXPECT_DOUBLE_EQ(__moore_covergroup_get_goal(cg), 0.0);

  __moore_covergroup_set_goal(cg, 150.0);
  EXPECT_DOUBLE_EQ(__moore_covergroup_get_goal(cg), 100.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CovergroupGoalMet) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp");

  // Set a goal of 50%
  __moore_covergroup_set_goal(cg, 50.0);

  // With no samples, goal should not be met
  EXPECT_FALSE(__moore_covergroup_goal_met(cg));

  // Sample a single value (100% coverage for single-value range)
  __moore_coverpoint_sample(cg, 0, 42);

  // Now goal should be met (100% >= 50%)
  EXPECT_TRUE(__moore_covergroup_goal_met(cg));

  __moore_covergroup_destroy(cg);
}

//===----------------------------------------------------------------------===//
// Total Coverage Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCoverageTest, TotalCoverage) {
  // Create two covergroups
  void *cg1 = __moore_covergroup_create("cg1", 1);
  void *cg2 = __moore_covergroup_create("cg2", 1);
  ASSERT_NE(cg1, nullptr);
  ASSERT_NE(cg2, nullptr);

  __moore_coverpoint_init(cg1, 0, "cp1");
  __moore_coverpoint_init(cg2, 0, "cp2");

  // Sample single values (each gets 100% coverage)
  __moore_coverpoint_sample(cg1, 0, 1);
  __moore_coverpoint_sample(cg2, 0, 2);

  // Total coverage should be 100% (average of two 100% covergroups)
  double total = __moore_coverage_get_total();
  EXPECT_DOUBLE_EQ(total, 100.0);

  __moore_covergroup_destroy(cg1);
  __moore_covergroup_destroy(cg2);
}

TEST(MooreRuntimeCoverageTest, NumCovergroups) {
  int32_t initialCount = __moore_coverage_get_num_covergroups();

  void *cg1 = __moore_covergroup_create("cg1", 1);
  EXPECT_EQ(__moore_coverage_get_num_covergroups(), initialCount + 1);

  void *cg2 = __moore_covergroup_create("cg2", 1);
  EXPECT_EQ(__moore_coverage_get_num_covergroups(), initialCount + 2);

  __moore_covergroup_destroy(cg1);
  EXPECT_EQ(__moore_coverage_get_num_covergroups(), initialCount + 1);

  __moore_covergroup_destroy(cg2);
  EXPECT_EQ(__moore_coverage_get_num_covergroups(), initialCount);
}

//===----------------------------------------------------------------------===//
// Explicit Bins Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCoverageTest, ExplicitBins) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  // Initialize with explicit bins
  MooreCoverageBin bins[] = {
    {"low", MOORE_BIN_RANGE, 0, 10, 0},
    {"mid", MOORE_BIN_RANGE, 11, 20, 0},
    {"high", MOORE_BIN_RANGE, 21, 30, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 3);

  // Sample values in different bins
  __moore_coverpoint_sample(cg, 0, 5);   // low bin
  __moore_coverpoint_sample(cg, 0, 15);  // mid bin

  // Check bin hits
  EXPECT_GE(__moore_coverpoint_get_bin_hits(cg, 0, 0), 1);  // low bin hit
  EXPECT_GE(__moore_coverpoint_get_bin_hits(cg, 0, 1), 1);  // mid bin hit
  EXPECT_EQ(__moore_coverpoint_get_bin_hits(cg, 0, 2), 0);  // high bin not hit

  // Coverage should be 2/3 = 66.67%
  double cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_GT(cov, 60.0);
  EXPECT_LT(cov, 70.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, AddBinDynamically) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp");

  // Add bins dynamically
  __moore_coverpoint_add_bin(cg, 0, "bin0", MOORE_BIN_VALUE, 0, 0);
  __moore_coverpoint_add_bin(cg, 0, "bin1", MOORE_BIN_VALUE, 1, 1);

  // Sample value 0
  __moore_coverpoint_sample(cg, 0, 0);

  // bin0 should be hit, bin1 should not
  EXPECT_GE(__moore_coverpoint_get_bin_hits(cg, 0, 0), 1);
  EXPECT_EQ(__moore_coverpoint_get_bin_hits(cg, 0, 1), 0);

  __moore_covergroup_destroy(cg);
}

//===----------------------------------------------------------------------===//
// HTML Report Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCoverageTest, HtmlReportGeneration) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "signal_a");
  __moore_coverpoint_init(cg, 1, "signal_b");

  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 0, 2);
  __moore_coverpoint_sample(cg, 1, 10);

  // Generate HTML report
  const char *filename = "/tmp/coverage_test_report.html";
  int32_t result = __moore_coverage_report_html(filename);
  EXPECT_EQ(result, 0);

  // Verify file was created (we won't parse HTML, just check creation)
  FILE *fp = std::fopen(filename, "r");
  EXPECT_NE(fp, nullptr);
  if (fp) {
    std::fclose(fp);
    std::remove(filename);
  }

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, HtmlReportNullFilename) {
  int32_t result = __moore_coverage_report_html(nullptr);
  EXPECT_NE(result, 0);
}

} // namespace
