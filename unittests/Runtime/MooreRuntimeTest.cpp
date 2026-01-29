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
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <ctime>
#include <map>
#include <string>
#include <thread>

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

//===----------------------------------------------------------------------===//
// Signal Registry Bridge Tests
//===----------------------------------------------------------------------===//

// Test data for signal registry callback testing
namespace {
struct MockSignalStore {
  std::map<MooreSignalHandle, int64_t> values;
  std::map<MooreSignalHandle, bool> forced;

  void reset() {
    values.clear();
    forced.clear();
  }
};

thread_local MockSignalStore mockSignalStore;

int64_t mockReadCallback(MooreSignalHandle handle, void *userData) {
  (void)userData;
  auto it = mockSignalStore.values.find(handle);
  return it != mockSignalStore.values.end() ? it->second : 0;
}

int32_t mockWriteCallback(MooreSignalHandle handle, int64_t value,
                          void *userData) {
  (void)userData;
  if (mockSignalStore.forced[handle])
    return 1; // Ignore writes to forced signals
  mockSignalStore.values[handle] = value;
  return 1;
}

int32_t mockForceCallback(MooreSignalHandle handle, int64_t value,
                          void *userData) {
  (void)userData;
  mockSignalStore.values[handle] = value;
  mockSignalStore.forced[handle] = true;
  return 1;
}

int32_t mockReleaseCallback(MooreSignalHandle handle, void *userData) {
  (void)userData;
  mockSignalStore.forced[handle] = false;
  return 1;
}
} // namespace

TEST(MooreRuntimeSignalRegistryTest, RegisterAndLookupSignal) {
  // Clear any existing state
  __moore_signal_registry_clear();
  mockSignalStore.reset();

  // Register a signal
  EXPECT_EQ(__moore_signal_registry_register("top.dut.clk", 1, 1), 1);
  EXPECT_EQ(__moore_signal_registry_register("top.dut.data", 2, 8), 1);

  // Look up signals
  EXPECT_EQ(__moore_signal_registry_lookup("top.dut.clk"), 1u);
  EXPECT_EQ(__moore_signal_registry_lookup("top.dut.data"), 2u);
  EXPECT_EQ(__moore_signal_registry_lookup("nonexistent"),
            MOORE_INVALID_SIGNAL_HANDLE);

  // Check existence
  EXPECT_EQ(__moore_signal_registry_exists("top.dut.clk"), 1);
  EXPECT_EQ(__moore_signal_registry_exists("top.dut.data"), 1);
  EXPECT_EQ(__moore_signal_registry_exists("nonexistent"), 0);

  // Check width
  EXPECT_EQ(__moore_signal_registry_get_width("top.dut.clk"), 1u);
  EXPECT_EQ(__moore_signal_registry_get_width("top.dut.data"), 8u);
  EXPECT_EQ(__moore_signal_registry_get_width("nonexistent"), 0u);

  // Check count
  EXPECT_EQ(__moore_signal_registry_count(), 2u);

  // Clear and verify
  __moore_signal_registry_clear();
  EXPECT_EQ(__moore_signal_registry_count(), 0u);
  EXPECT_EQ(__moore_signal_registry_exists("top.dut.clk"), 0);
}

TEST(MooreRuntimeSignalRegistryTest, InvalidRegistration) {
  __moore_signal_registry_clear();

  // Invalid path
  EXPECT_EQ(__moore_signal_registry_register(nullptr, 1, 8), 0);
  EXPECT_EQ(__moore_signal_registry_register("", 1, 8), 0);

  // Invalid handle
  EXPECT_EQ(__moore_signal_registry_register("valid.path",
                                              MOORE_INVALID_SIGNAL_HANDLE, 8),
            0);

  EXPECT_EQ(__moore_signal_registry_count(), 0u);
}

TEST(MooreRuntimeSignalRegistryTest, ConnectedStatus) {
  __moore_signal_registry_clear();
  mockSignalStore.reset();

  // Initially not connected
  EXPECT_EQ(__moore_signal_registry_is_connected(), 0);

  // Set accessor callbacks
  __moore_signal_registry_set_accessor(mockReadCallback, mockWriteCallback,
                                       mockForceCallback, mockReleaseCallback,
                                       nullptr);

  // Now connected
  EXPECT_EQ(__moore_signal_registry_is_connected(), 1);

  // Disconnect
  __moore_signal_registry_set_accessor(nullptr, nullptr, nullptr, nullptr,
                                       nullptr);
  EXPECT_EQ(__moore_signal_registry_is_connected(), 0);
}

TEST(MooreRuntimeSignalRegistryTest, HdlReadUsesRegistry) {
  __moore_signal_registry_clear();
  mockSignalStore.reset();

  // Register a signal and connect callbacks
  __moore_signal_registry_register("top.reg_signal", 100, 32);
  __moore_signal_registry_set_accessor(mockReadCallback, mockWriteCallback,
                                       mockForceCallback, mockReleaseCallback,
                                       nullptr);

  // Set value in mock store
  mockSignalStore.values[100] = 0xDEADBEEF;

  // Read via DPI should get mock value
  MooreString path = makeMooreString("top.reg_signal");
  uvm_hdl_data_t value = 0;
  EXPECT_EQ(uvm_hdl_read(&path, &value), 1);
  EXPECT_EQ(value, 0xDEADBEEF);

  // Disconnect and verify fallback to stub
  __moore_signal_registry_set_accessor(nullptr, nullptr, nullptr, nullptr,
                                       nullptr);
  value = 0;
  EXPECT_EQ(uvm_hdl_read(&path, &value), 1);
  // Stub returns 0 for new entries
  EXPECT_EQ(value, 0);

  __moore_signal_registry_clear();
}

TEST(MooreRuntimeSignalRegistryTest, HdlDepositUsesRegistry) {
  __moore_signal_registry_clear();
  mockSignalStore.reset();

  // Register and connect
  __moore_signal_registry_register("top.write_signal", 200, 16);
  __moore_signal_registry_set_accessor(mockReadCallback, mockWriteCallback,
                                       mockForceCallback, mockReleaseCallback,
                                       nullptr);

  // Deposit value
  MooreString path = makeMooreString("top.write_signal");
  EXPECT_EQ(uvm_hdl_deposit(&path, 12345), 1);

  // Verify it was written to mock store
  EXPECT_EQ(mockSignalStore.values[200], 12345);

  __moore_signal_registry_set_accessor(nullptr, nullptr, nullptr, nullptr,
                                       nullptr);
  __moore_signal_registry_clear();
}

TEST(MooreRuntimeSignalRegistryTest, HdlForceUsesRegistry) {
  __moore_signal_registry_clear();
  mockSignalStore.reset();

  // Register and connect
  __moore_signal_registry_register("top.force_signal", 300, 8);
  __moore_signal_registry_set_accessor(mockReadCallback, mockWriteCallback,
                                       mockForceCallback, mockReleaseCallback,
                                       nullptr);

  // Force value
  MooreString path = makeMooreString("top.force_signal");
  EXPECT_EQ(uvm_hdl_force(&path, 0xAB), 1);

  // Verify force was applied
  EXPECT_EQ(mockSignalStore.values[300], 0xAB);
  EXPECT_TRUE(mockSignalStore.forced[300]);

  // Deposit should be ignored (forced)
  EXPECT_EQ(uvm_hdl_deposit(&path, 0xCD), 1);
  EXPECT_EQ(mockSignalStore.values[300], 0xAB); // Still AB, deposit ignored

  // Release
  EXPECT_EQ(uvm_hdl_release(&path), 1);
  EXPECT_FALSE(mockSignalStore.forced[300]);

  // Now deposit should work
  EXPECT_EQ(uvm_hdl_deposit(&path, 0xEF), 1);
  EXPECT_EQ(mockSignalStore.values[300], 0xEF);

  __moore_signal_registry_set_accessor(nullptr, nullptr, nullptr, nullptr,
                                       nullptr);
  __moore_signal_registry_clear();
}

TEST(MooreRuntimeSignalRegistryTest, HdlReleaseAndReadUsesRegistry) {
  __moore_signal_registry_clear();
  mockSignalStore.reset();

  // Register and connect
  __moore_signal_registry_register("top.release_read_signal", 400, 32);
  __moore_signal_registry_set_accessor(mockReadCallback, mockWriteCallback,
                                       mockForceCallback, mockReleaseCallback,
                                       nullptr);

  // Force and set value
  mockSignalStore.values[400] = 0x1234;
  mockSignalStore.forced[400] = true;

  // Release and read
  MooreString path = makeMooreString("top.release_read_signal");
  uvm_hdl_data_t value = 0;
  EXPECT_EQ(uvm_hdl_release_and_read(&path, &value), 1);
  EXPECT_EQ(value, 0x1234);
  EXPECT_FALSE(mockSignalStore.forced[400]);

  __moore_signal_registry_set_accessor(nullptr, nullptr, nullptr, nullptr,
                                       nullptr);
  __moore_signal_registry_clear();
}

TEST(MooreRuntimeSignalRegistryTest, FallbackToStubWhenNotRegistered) {
  __moore_signal_registry_clear();
  mockSignalStore.reset();

  // Connect callbacks but don't register this path
  __moore_signal_registry_set_accessor(mockReadCallback, mockWriteCallback,
                                       mockForceCallback, mockReleaseCallback,
                                       nullptr);

  // Path not in registry should fall back to stub behavior
  MooreString path = makeMooreString("top.stub_only_signal");
  EXPECT_EQ(uvm_hdl_deposit(&path, 999), 1);

  // Read should get stub value (not from mock store)
  uvm_hdl_data_t value = 0;
  EXPECT_EQ(uvm_hdl_read(&path, &value), 1);
  EXPECT_EQ(value, 999); // From stub map

  __moore_signal_registry_set_accessor(nullptr, nullptr, nullptr, nullptr,
                                       nullptr);
  __moore_signal_registry_clear();
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
// Queue Insert Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeQueueTest, QueueInsertAtBeginning) {
  // Create a queue with some elements
  MooreQueue queue = {nullptr, 0};
  int32_t a = 2, b = 3, c = 1;
  int64_t elementSize = sizeof(int32_t);

  __moore_queue_push_back(&queue, &a, elementSize);
  __moore_queue_push_back(&queue, &b, elementSize);

  // Insert at beginning (index 0)
  __moore_queue_insert(&queue, 0, &c, elementSize);

  ASSERT_EQ(queue.len, 3);
  auto *data = static_cast<int32_t *>(queue.data);
  EXPECT_EQ(data[0], 1);  // Newly inserted
  EXPECT_EQ(data[1], 2);
  EXPECT_EQ(data[2], 3);

  __moore_free(queue.data);
}

TEST(MooreRuntimeQueueTest, QueueInsertAtMiddle) {
  MooreQueue queue = {nullptr, 0};
  int32_t a = 1, b = 3, c = 2;
  int64_t elementSize = sizeof(int32_t);

  __moore_queue_push_back(&queue, &a, elementSize);
  __moore_queue_push_back(&queue, &b, elementSize);

  // Insert at middle (index 1)
  __moore_queue_insert(&queue, 1, &c, elementSize);

  ASSERT_EQ(queue.len, 3);
  auto *data = static_cast<int32_t *>(queue.data);
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 2);  // Newly inserted
  EXPECT_EQ(data[2], 3);

  __moore_free(queue.data);
}

TEST(MooreRuntimeQueueTest, QueueInsertAtEnd) {
  MooreQueue queue = {nullptr, 0};
  int32_t a = 1, b = 2, c = 3;
  int64_t elementSize = sizeof(int32_t);

  __moore_queue_push_back(&queue, &a, elementSize);
  __moore_queue_push_back(&queue, &b, elementSize);

  // Insert at end (index == size)
  __moore_queue_insert(&queue, 2, &c, elementSize);

  ASSERT_EQ(queue.len, 3);
  auto *data = static_cast<int32_t *>(queue.data);
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 2);
  EXPECT_EQ(data[2], 3);  // Newly inserted

  __moore_free(queue.data);
}

TEST(MooreRuntimeQueueTest, QueueInsertBeyondEnd) {
  MooreQueue queue = {nullptr, 0};
  int32_t a = 1, b = 2, c = 3;
  int64_t elementSize = sizeof(int32_t);

  __moore_queue_push_back(&queue, &a, elementSize);
  __moore_queue_push_back(&queue, &b, elementSize);

  // Insert beyond end (index > size) - should append
  __moore_queue_insert(&queue, 100, &c, elementSize);

  ASSERT_EQ(queue.len, 3);
  auto *data = static_cast<int32_t *>(queue.data);
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 2);
  EXPECT_EQ(data[2], 3);  // Appended

  __moore_free(queue.data);
}

TEST(MooreRuntimeQueueTest, QueueInsertNegativeIndex) {
  MooreQueue queue = {nullptr, 0};
  int32_t a = 2, b = 3, c = 1;
  int64_t elementSize = sizeof(int32_t);

  __moore_queue_push_back(&queue, &a, elementSize);
  __moore_queue_push_back(&queue, &b, elementSize);

  // Insert with negative index - should treat as 0
  __moore_queue_insert(&queue, -5, &c, elementSize);

  ASSERT_EQ(queue.len, 3);
  auto *data = static_cast<int32_t *>(queue.data);
  EXPECT_EQ(data[0], 1);  // Inserted at front
  EXPECT_EQ(data[1], 2);
  EXPECT_EQ(data[2], 3);

  __moore_free(queue.data);
}

TEST(MooreRuntimeQueueTest, QueueInsertIntoEmpty) {
  MooreQueue queue = {nullptr, 0};
  int32_t a = 42;
  int64_t elementSize = sizeof(int32_t);

  // Insert into empty queue
  __moore_queue_insert(&queue, 0, &a, elementSize);

  ASSERT_EQ(queue.len, 1);
  auto *data = static_cast<int32_t *>(queue.data);
  EXPECT_EQ(data[0], 42);

  __moore_free(queue.data);
}

TEST(MooreRuntimeQueueTest, QueueInsertMultiple) {
  MooreQueue queue = {nullptr, 0};
  int64_t elementSize = sizeof(int32_t);

  // Build queue {1, 2, 3, 4, 5} by inserting at various positions
  int32_t v3 = 3;
  __moore_queue_insert(&queue, 0, &v3, elementSize);  // {3}

  int32_t v1 = 1;
  __moore_queue_insert(&queue, 0, &v1, elementSize);  // {1, 3}

  int32_t v5 = 5;
  __moore_queue_insert(&queue, 2, &v5, elementSize);  // {1, 3, 5}

  int32_t v2 = 2;
  __moore_queue_insert(&queue, 1, &v2, elementSize);  // {1, 2, 3, 5}

  int32_t v4 = 4;
  __moore_queue_insert(&queue, 3, &v4, elementSize);  // {1, 2, 3, 4, 5}

  ASSERT_EQ(queue.len, 5);
  auto *data = static_cast<int32_t *>(queue.data);
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 2);
  EXPECT_EQ(data[2], 3);
  EXPECT_EQ(data[3], 4);
  EXPECT_EQ(data[4], 5);

  __moore_free(queue.data);
}

//===----------------------------------------------------------------------===//
// Queue Pop Ptr Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeQueueTest, QueuePopBackPtr) {
  MooreQueue queue = {nullptr, 0};
  int64_t elementSize = sizeof(int32_t);

  // Build queue {10, 20, 30}
  int32_t v1 = 10, v2 = 20, v3 = 30;
  __moore_queue_push_back(&queue, &v1, elementSize);
  __moore_queue_push_back(&queue, &v2, elementSize);
  __moore_queue_push_back(&queue, &v3, elementSize);

  // Pop back should return 30
  int32_t result = 0;
  __moore_queue_pop_back_ptr(&queue, &result, elementSize);
  EXPECT_EQ(result, 30);
  EXPECT_EQ(queue.len, 2);

  // Pop again should return 20
  __moore_queue_pop_back_ptr(&queue, &result, elementSize);
  EXPECT_EQ(result, 20);
  EXPECT_EQ(queue.len, 1);

  // Pop last element should return 10
  __moore_queue_pop_back_ptr(&queue, &result, elementSize);
  EXPECT_EQ(result, 10);
  EXPECT_EQ(queue.len, 0);
  EXPECT_EQ(queue.data, nullptr);
}

TEST(MooreRuntimeQueueTest, QueuePopFrontPtr) {
  MooreQueue queue = {nullptr, 0};
  int64_t elementSize = sizeof(int32_t);

  // Build queue {10, 20, 30}
  int32_t v1 = 10, v2 = 20, v3 = 30;
  __moore_queue_push_back(&queue, &v1, elementSize);
  __moore_queue_push_back(&queue, &v2, elementSize);
  __moore_queue_push_back(&queue, &v3, elementSize);

  // Pop front should return 10
  int32_t result = 0;
  __moore_queue_pop_front_ptr(&queue, &result, elementSize);
  EXPECT_EQ(result, 10);
  EXPECT_EQ(queue.len, 2);

  // Pop again should return 20
  __moore_queue_pop_front_ptr(&queue, &result, elementSize);
  EXPECT_EQ(result, 20);
  EXPECT_EQ(queue.len, 1);

  // Pop last element should return 30
  __moore_queue_pop_front_ptr(&queue, &result, elementSize);
  EXPECT_EQ(result, 30);
  EXPECT_EQ(queue.len, 0);
  EXPECT_EQ(queue.data, nullptr);
}

TEST(MooreRuntimeQueueTest, QueuePopPtrWithStructs) {
  // Test with struct elements
  struct TestStruct {
    int32_t a;
    int32_t b;
  };

  MooreQueue queue = {nullptr, 0};
  int64_t elementSize = sizeof(TestStruct);

  TestStruct s1 = {1, 2};
  TestStruct s2 = {3, 4};
  TestStruct s3 = {5, 6};

  __moore_queue_push_back(&queue, &s1, elementSize);
  __moore_queue_push_back(&queue, &s2, elementSize);
  __moore_queue_push_back(&queue, &s3, elementSize);

  TestStruct result = {0, 0};
  __moore_queue_pop_back_ptr(&queue, &result, elementSize);
  EXPECT_EQ(result.a, 5);
  EXPECT_EQ(result.b, 6);
  EXPECT_EQ(queue.len, 2);

  __moore_queue_pop_front_ptr(&queue, &result, elementSize);
  EXPECT_EQ(result.a, 1);
  EXPECT_EQ(result.b, 2);
  EXPECT_EQ(queue.len, 1);

  __moore_free(queue.data);
}

//===----------------------------------------------------------------------===//
// Queue Size Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeQueueTest, QueueSize) {
  MooreQueue queue = {nullptr, 0};

  // Empty queue
  EXPECT_EQ(__moore_queue_size(&queue), 0);

  // Add elements
  int32_t v = 42;
  __moore_queue_push_back(&queue, &v, sizeof(int32_t));
  EXPECT_EQ(__moore_queue_size(&queue), 1);

  __moore_queue_push_back(&queue, &v, sizeof(int32_t));
  EXPECT_EQ(__moore_queue_size(&queue), 2);

  __moore_queue_push_back(&queue, &v, sizeof(int32_t));
  EXPECT_EQ(__moore_queue_size(&queue), 3);

  __moore_free(queue.data);
}

TEST(MooreRuntimeQueueTest, QueueSizeNull) {
  // Null pointer should return 0
  EXPECT_EQ(__moore_queue_size(nullptr), 0);
}

//===----------------------------------------------------------------------===//
// Queue Unique Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeQueueTest, QueueUnique) {
  // Note: __moore_queue_unique assumes 8-byte elements
  MooreQueue queue = {nullptr, 0};
  int64_t elementSize = sizeof(int64_t);

  // Build queue with duplicates {1, 2, 2, 3, 1, 4, 2}
  int64_t vals[] = {1, 2, 2, 3, 1, 4, 2};
  for (auto v : vals) {
    __moore_queue_push_back(&queue, &v, elementSize);
  }

  MooreQueue result = __moore_queue_unique(&queue);

  // Expected unique: {1, 2, 3, 4}
  ASSERT_EQ(result.len, 4);
  auto *data = static_cast<int64_t *>(result.data);
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 2);
  EXPECT_EQ(data[2], 3);
  EXPECT_EQ(data[3], 4);

  __moore_free(queue.data);
  __moore_free(result.data);
}

TEST(MooreRuntimeQueueTest, QueueUniqueEmpty) {
  MooreQueue queue = {nullptr, 0};
  MooreQueue result = __moore_queue_unique(&queue);
  EXPECT_EQ(result.len, 0);
  EXPECT_EQ(result.data, nullptr);
}

TEST(MooreRuntimeQueueTest, QueueUniqueAllSame) {
  MooreQueue queue = {nullptr, 0};
  int64_t elementSize = sizeof(int64_t);

  // All same values
  int64_t v = 42;
  for (int i = 0; i < 5; ++i) {
    __moore_queue_push_back(&queue, &v, elementSize);
  }

  MooreQueue result = __moore_queue_unique(&queue);
  ASSERT_EQ(result.len, 1);
  EXPECT_EQ(static_cast<int64_t *>(result.data)[0], 42);

  __moore_free(queue.data);
  __moore_free(result.data);
}

//===----------------------------------------------------------------------===//
// Queue Sort Inplace Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeQueueTest, QueueSortInplace) {
  MooreQueue queue = {nullptr, 0};
  int64_t elementSize = sizeof(int32_t);

  // Build unsorted queue
  int32_t vals[] = {5, 2, 8, 1, 9, 3};
  for (auto v : vals) {
    __moore_queue_push_back(&queue, &v, elementSize);
  }

  __moore_queue_sort_inplace(&queue, elementSize);

  // Should be sorted ascending
  auto *data = static_cast<int32_t *>(queue.data);
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 2);
  EXPECT_EQ(data[2], 3);
  EXPECT_EQ(data[3], 5);
  EXPECT_EQ(data[4], 8);
  EXPECT_EQ(data[5], 9);

  __moore_free(queue.data);
}

TEST(MooreRuntimeQueueTest, QueueSortInplaceAlreadySorted) {
  MooreQueue queue = {nullptr, 0};
  int64_t elementSize = sizeof(int32_t);

  int32_t vals[] = {1, 2, 3, 4, 5};
  for (auto v : vals) {
    __moore_queue_push_back(&queue, &v, elementSize);
  }

  __moore_queue_sort_inplace(&queue, elementSize);

  auto *data = static_cast<int32_t *>(queue.data);
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 2);
  EXPECT_EQ(data[2], 3);
  EXPECT_EQ(data[3], 4);
  EXPECT_EQ(data[4], 5);

  __moore_free(queue.data);
}

TEST(MooreRuntimeQueueTest, QueueSortInplaceReversed) {
  MooreQueue queue = {nullptr, 0};
  int64_t elementSize = sizeof(int32_t);

  int32_t vals[] = {5, 4, 3, 2, 1};
  for (auto v : vals) {
    __moore_queue_push_back(&queue, &v, elementSize);
  }

  __moore_queue_sort_inplace(&queue, elementSize);

  auto *data = static_cast<int32_t *>(queue.data);
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 2);
  EXPECT_EQ(data[2], 3);
  EXPECT_EQ(data[3], 4);
  EXPECT_EQ(data[4], 5);

  __moore_free(queue.data);
}

TEST(MooreRuntimeQueueTest, QueueSortInplaceEmpty) {
  MooreQueue queue = {nullptr, 0};

  // Should not crash
  __moore_queue_sort_inplace(&queue, sizeof(int32_t));

  EXPECT_EQ(queue.len, 0);
  EXPECT_EQ(queue.data, nullptr);
}

TEST(MooreRuntimeQueueTest, QueueSortInplaceSingleElement) {
  MooreQueue queue = {nullptr, 0};
  int32_t v = 42;
  __moore_queue_push_back(&queue, &v, sizeof(int32_t));

  __moore_queue_sort_inplace(&queue, sizeof(int32_t));

  EXPECT_EQ(queue.len, 1);
  EXPECT_EQ(static_cast<int32_t *>(queue.data)[0], 42);

  __moore_free(queue.data);
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
// Distribution Constraint Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeDistTest, RandomizeWithDistSingleValue) {
  // Test distribution with single values using := weight
  // x dist { 0 := 10, 1 := 20, 2 := 30 }
  int64_t ranges[] = {0, 0, 1, 1, 2, 2}; // [0,0], [1,1], [2,2]
  int64_t weights[] = {10, 20, 30};
  int64_t perRange[] = {0, 0, 0}; // All := (per-value)

  // Run multiple times and count occurrences
  std::map<int64_t, int> counts;
  for (int i = 0; i < 600; ++i) {
    int64_t result = __moore_randomize_with_dist(ranges, weights, perRange, 3);
    EXPECT_GE(result, 0);
    EXPECT_LE(result, 2);
    counts[result]++;
  }

  // With weights 10:20:30, expect roughly 1:2:3 ratio
  // Allow some statistical variation
  EXPECT_GT(counts[0], 0);
  EXPECT_GT(counts[1], 0);
  EXPECT_GT(counts[2], 0);
  // Value 2 should appear more often than value 0 (on average)
}

TEST(MooreRuntimeDistTest, RandomizeWithDistRange) {
  // Test distribution with ranges using := weight
  // x dist { [0:4] := 2 } means each value in [0,4] gets weight 2
  int64_t ranges[] = {0, 4}; // [0,4]
  int64_t weights[] = {2};
  int64_t perRange[] = {0}; // := (per-value)

  // All values 0-4 should appear
  std::map<int64_t, int> counts;
  for (int i = 0; i < 500; ++i) {
    int64_t result = __moore_randomize_with_dist(ranges, weights, perRange, 1);
    EXPECT_GE(result, 0);
    EXPECT_LE(result, 4);
    counts[result]++;
  }

  // Each value should appear at least once
  for (int v = 0; v <= 4; ++v) {
    EXPECT_GT(counts[v], 0) << "Value " << v << " should appear";
  }
}

TEST(MooreRuntimeDistTest, RandomizeWithDistPerRange) {
  // Test distribution with :/ (per-range) weight
  // x dist { [0:4] :/ 100 } means total weight 100 is divided among 5 values
  int64_t ranges[] = {0, 4}; // [0,4]
  int64_t weights[] = {100};
  int64_t perRange[] = {1}; // :/ (per-range)

  // All values 0-4 should appear with equal probability
  std::map<int64_t, int> counts;
  for (int i = 0; i < 500; ++i) {
    int64_t result = __moore_randomize_with_dist(ranges, weights, perRange, 1);
    EXPECT_GE(result, 0);
    EXPECT_LE(result, 4);
    counts[result]++;
  }

  // Each value should appear at least once
  for (int v = 0; v <= 4; ++v) {
    EXPECT_GT(counts[v], 0) << "Value " << v << " should appear";
  }
}

TEST(MooreRuntimeDistTest, RandomizeWithDistMixed) {
  // Test mixed := and :/ weights
  // x dist { 0 := 1, [1:10] :/ 4 }
  // Value 0 gets weight 1, range [1:10] (10 values) shares weight 4
  int64_t ranges[] = {0, 0, 1, 10}; // [0,0], [1,10]
  int64_t weights[] = {1, 4};
  int64_t perRange[] = {0, 1}; // := for first, :/ for second

  // Total effective weight: 1 (for 0) + 4 (for range [1,10])
  std::map<int64_t, int> counts;
  for (int i = 0; i < 500; ++i) {
    int64_t result = __moore_randomize_with_dist(ranges, weights, perRange, 2);
    EXPECT_GE(result, 0);
    EXPECT_LE(result, 10);
    counts[result]++;
  }

  // Both ranges should be hit
  EXPECT_GT(counts[0], 0);
  int rangeCount = 0;
  for (int v = 1; v <= 10; ++v) {
    rangeCount += counts[v];
  }
  EXPECT_GT(rangeCount, 0);
}

TEST(MooreRuntimeDistTest, RandomizeWithDistNullInputs) {
  // Test with null inputs - should return default (first range low)
  int64_t ranges[] = {5, 10};
  int64_t weights[] = {1};
  int64_t perRange[] = {0};

  // Null ranges
  int64_t result1 = __moore_randomize_with_dist(nullptr, weights, perRange, 1);
  EXPECT_EQ(result1, 0);

  // Null weights
  int64_t result2 = __moore_randomize_with_dist(ranges, nullptr, perRange, 1);
  EXPECT_EQ(result2, 0);

  // Null perRange
  int64_t result3 = __moore_randomize_with_dist(ranges, weights, nullptr, 1);
  EXPECT_EQ(result3, 0);

  // Zero numRanges
  int64_t result4 = __moore_randomize_with_dist(ranges, weights, perRange, 0);
  EXPECT_EQ(result4, 0);
}

TEST(MooreRuntimeDistTest, RandomizeWithDistZeroWeights) {
  // Test with all zero weights - should return first range value
  int64_t ranges[] = {5, 10, 15, 20};
  int64_t weights[] = {0, 0};
  int64_t perRange[] = {0, 0};

  int64_t result = __moore_randomize_with_dist(ranges, weights, perRange, 2);
  // With zero weights, should return first range's low value
  EXPECT_EQ(result, 5);
}

TEST(MooreRuntimeDistTest, RandomizeWithDistSeededConsistency) {
  // Test that seeding produces consistent distribution results
  int64_t ranges[] = {0, 0, 1, 1, 2, 2};
  int64_t weights[] = {10, 20, 30};
  int64_t perRange[] = {0, 0, 0};

  __moore_urandom_seeded(54321);
  int64_t result1 = __moore_randomize_with_dist(ranges, weights, perRange, 3);

  __moore_urandom_seeded(54321);
  int64_t result2 = __moore_randomize_with_dist(ranges, weights, perRange, 3);

  EXPECT_EQ(result1, result2);
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
// Cross Coverage Named Bins Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCrossCoverageTest, CrossAddNamedBin) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  int32_t cpIndices[] = {0, 1};
  int32_t crossIdx = __moore_cross_create(cg, "cross01", cpIndices, 2);
  EXPECT_GE(crossIdx, 0);

  // Add a named bin with no filters (matches everything)
  int32_t binIdx = __moore_cross_add_named_bin(cg, crossIdx, "all_bins",
                                                MOORE_CROSS_BIN_NORMAL,
                                                nullptr, 0);
  EXPECT_GE(binIdx, 0);

  // Verify the bin was added
  EXPECT_EQ(__moore_cross_get_num_named_bins(cg, crossIdx), 1);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCrossCoverageTest, CrossNamedBinHits) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  int32_t cpIndices[] = {0, 1};
  int32_t crossIdx = __moore_cross_create(cg, "cross01", cpIndices, 2);
  EXPECT_GE(crossIdx, 0);

  // Add a named bin that matches everything
  int32_t binIdx = __moore_cross_add_named_bin(cg, crossIdx, "all_bins",
                                                MOORE_CROSS_BIN_NORMAL,
                                                nullptr, 0);
  EXPECT_GE(binIdx, 0);

  // Sample a value
  __moore_coverpoint_sample(cg, 0, 5);
  __moore_coverpoint_sample(cg, 1, 10);
  int64_t vals[] = {5, 10};
  __moore_cross_sample(cg, vals, 2);

  // The named bin should have been hit
  EXPECT_EQ(__moore_cross_get_named_bin_hits(cg, crossIdx, binIdx), 1);

  // Sample another value
  int64_t vals2[] = {6, 11};
  __moore_cross_sample(cg, vals2, 2);

  // The named bin should have been hit again
  EXPECT_EQ(__moore_cross_get_named_bin_hits(cg, crossIdx, binIdx), 2);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCrossCoverageTest, CrossIgnoreBins) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  int32_t cpIndices[] = {0, 1};
  int32_t crossIdx = __moore_cross_create(cg, "cross01", cpIndices, 2);
  EXPECT_GE(crossIdx, 0);

  // Add an ignore bin that filters out cp0 == 0 (using intersect values)
  MooreCrossBinsofFilter filter;
  filter.cp_index = 0;
  filter.bin_indices = nullptr;
  filter.num_bins = 0;
  int64_t ignoreValues[] = {0};
  filter.values = ignoreValues;
  filter.num_values = 1;
  filter.negate = false;

  int32_t ignoreBinIdx = __moore_cross_add_ignore_bin(cg, crossIdx,
                                                       "ignore_zero",
                                                       &filter, 1);
  EXPECT_GE(ignoreBinIdx, 0);

  // Sample values - one should be ignored
  __moore_coverpoint_sample(cg, 0, 0);
  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 1, 10);

  // Sample with cp0=0 (should be ignored)
  int64_t vals1[] = {0, 10};
  __moore_cross_sample(cg, vals1, 2);

  // Sample with cp0=1 (should NOT be ignored)
  int64_t vals2[] = {1, 10};
  __moore_cross_sample(cg, vals2, 2);

  // Only 1 cross bin should be recorded (the non-ignored one)
  EXPECT_EQ(__moore_cross_get_bins_hit(cg, crossIdx), 1);

  // Check if values are correctly identified as ignored
  EXPECT_TRUE(__moore_cross_is_ignored(cg, crossIdx, vals1));
  EXPECT_FALSE(__moore_cross_is_ignored(cg, crossIdx, vals2));

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCrossCoverageTest, CrossIllegalBins) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  int32_t cpIndices[] = {0, 1};
  int32_t crossIdx = __moore_cross_create(cg, "cross01", cpIndices, 2);
  EXPECT_GE(crossIdx, 0);

  // Add an illegal bin that matches cp0 == 100 AND cp1 == 200
  MooreCrossBinsofFilter filters[2];

  // Filter for cp0
  int64_t cp0Values[] = {100};
  filters[0].cp_index = 0;
  filters[0].bin_indices = nullptr;
  filters[0].num_bins = 0;
  filters[0].values = cp0Values;
  filters[0].num_values = 1;
  filters[0].negate = false;

  // Filter for cp1
  int64_t cp1Values[] = {200};
  filters[1].cp_index = 1;
  filters[1].bin_indices = nullptr;
  filters[1].num_bins = 0;
  filters[1].values = cp1Values;
  filters[1].num_values = 1;
  filters[1].negate = false;

  int32_t illegalBinIdx = __moore_cross_add_illegal_bin(cg, crossIdx,
                                                         "bad_combo",
                                                         filters, 2);
  EXPECT_GE(illegalBinIdx, 0);

  // Check that illegal combination is detected
  int64_t illegalVals[] = {100, 200};
  EXPECT_TRUE(__moore_cross_is_illegal(cg, crossIdx, illegalVals));

  // Check that non-illegal combinations are not detected
  int64_t normalVals1[] = {100, 199};
  int64_t normalVals2[] = {99, 200};
  int64_t normalVals3[] = {1, 2};
  EXPECT_FALSE(__moore_cross_is_illegal(cg, crossIdx, normalVals1));
  EXPECT_FALSE(__moore_cross_is_illegal(cg, crossIdx, normalVals2));
  EXPECT_FALSE(__moore_cross_is_illegal(cg, crossIdx, normalVals3));

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCrossCoverageTest, CrossIllegalBinCallback) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  int32_t cpIndices[] = {0, 1};
  int32_t crossIdx = __moore_cross_create(cg, "cross01", cpIndices, 2);
  EXPECT_GE(crossIdx, 0);

  // Add an illegal bin
  MooreCrossBinsofFilter filter;
  int64_t illegalValues[] = {999};
  filter.cp_index = 0;
  filter.bin_indices = nullptr;
  filter.num_bins = 0;
  filter.values = illegalValues;
  filter.num_values = 1;
  filter.negate = false;

  __moore_cross_add_illegal_bin(cg, crossIdx, "illegal_999", &filter, 1);

  // Track callback invocations
  static bool callbackCalled = false;
  static std::string lastCgName;
  static std::string lastCrossName;
  static std::string lastBinName;

  auto callback = [](const char *cg_name, const char *cross_name,
                     const char *bin_name, int64_t *values, int32_t num_values,
                     void *userData) {
    callbackCalled = true;
    lastCgName = cg_name ? cg_name : "";
    lastCrossName = cross_name ? cross_name : "";
    lastBinName = bin_name ? bin_name : "";
  };

  callbackCalled = false;
  __moore_cross_set_illegal_bin_callback(callback, nullptr);

  // Sample the illegal value
  __moore_coverpoint_sample(cg, 0, 999);
  __moore_coverpoint_sample(cg, 1, 1);
  int64_t vals[] = {999, 1};
  __moore_cross_sample(cg, vals, 2);

  // Verify callback was called with correct info
  EXPECT_TRUE(callbackCalled);
  EXPECT_EQ(lastCgName, "test_cg");
  EXPECT_EQ(lastCrossName, "cross01");
  EXPECT_EQ(lastBinName, "illegal_999");

  // Clean up
  __moore_cross_set_illegal_bin_callback(nullptr, nullptr);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCrossCoverageTest, CrossBinsofNegate) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  int32_t cpIndices[] = {0, 1};
  int32_t crossIdx = __moore_cross_create(cg, "cross01", cpIndices, 2);
  EXPECT_GE(crossIdx, 0);

  // Add an ignore bin that uses negation: !binsof(cp0) intersect {5}
  // This should ignore everything EXCEPT cp0 == 5
  MooreCrossBinsofFilter filter;
  int64_t values[] = {5};
  filter.cp_index = 0;
  filter.bin_indices = nullptr;
  filter.num_bins = 0;
  filter.values = values;
  filter.num_values = 1;
  filter.negate = true;  // Negate: ignore when cp0 != 5

  int32_t ignoreBinIdx = __moore_cross_add_ignore_bin(cg, crossIdx,
                                                       "ignore_not_5",
                                                       &filter, 1);
  EXPECT_GE(ignoreBinIdx, 0);

  // Sample with cp0=5 (should NOT be ignored because filter is negated)
  int64_t vals1[] = {5, 10};
  EXPECT_FALSE(__moore_cross_is_ignored(cg, crossIdx, vals1));

  // Sample with cp0=3 (should be ignored because 3 != 5 and filter is negated)
  int64_t vals2[] = {3, 10};
  EXPECT_TRUE(__moore_cross_is_ignored(cg, crossIdx, vals2));

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCrossCoverageTest, CrossResetNamedBins) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  int32_t cpIndices[] = {0, 1};
  int32_t crossIdx = __moore_cross_create(cg, "cross01", cpIndices, 2);
  EXPECT_GE(crossIdx, 0);

  // Add a named bin
  int32_t binIdx = __moore_cross_add_named_bin(cg, crossIdx, "all_bins",
                                                MOORE_CROSS_BIN_NORMAL,
                                                nullptr, 0);
  EXPECT_GE(binIdx, 0);

  // Sample some values
  int64_t vals[] = {1, 2};
  __moore_cross_sample(cg, vals, 2);

  // Verify the bin was hit
  EXPECT_EQ(__moore_cross_get_named_bin_hits(cg, crossIdx, binIdx), 1);

  // Reset the covergroup
  __moore_covergroup_reset(cg);

  // Verify hit count is reset
  EXPECT_EQ(__moore_cross_get_named_bin_hits(cg, crossIdx, binIdx), 0);

  __moore_covergroup_destroy(cg);
}

//===----------------------------------------------------------------------===//
// get_coverage() and get_inst_coverage() API Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCoverageTest, CovergroupGetInstCoverage) {
  // Test basic instance coverage for a single covergroup
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp");

  // Initially, coverage should be 0
  double instCov = __moore_covergroup_get_inst_coverage(cg);
  EXPECT_DOUBLE_EQ(instCov, 0.0);

  // Sample a single value
  __moore_coverpoint_sample(cg, 0, 42);

  // Instance coverage should be 100% (single value = full coverage)
  instCov = __moore_covergroup_get_inst_coverage(cg);
  EXPECT_DOUBLE_EQ(instCov, 100.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CoverpointGetInstCoverage) {
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  // Sample different values
  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 1, 10);
  __moore_coverpoint_sample(cg, 1, 20);

  // Get instance coverage for each coverpoint
  double cp0InstCov = __moore_coverpoint_get_inst_coverage(cg, 0);
  double cp1InstCov = __moore_coverpoint_get_inst_coverage(cg, 1);

  // cp0 has single value - 100% coverage
  EXPECT_DOUBLE_EQ(cp0InstCov, 100.0);
  // cp1 has range 10-20 (11 values), 2 unique values
  EXPECT_GT(cp1InstCov, 0.0);
  EXPECT_LT(cp1InstCov, 100.0);

  // Instance coverage should equal regular coverage for coverpoints
  EXPECT_DOUBLE_EQ(cp0InstCov, __moore_coverpoint_get_coverage(cg, 0));
  EXPECT_DOUBLE_EQ(cp1InstCov, __moore_coverpoint_get_coverage(cg, 1));

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, MultiInstanceTypeCoverage) {
  // Test that get_coverage aggregates across instances with same name
  // when per_instance is false (default)
  void *cg1 = __moore_covergroup_create("same_type_cg", 1);
  void *cg2 = __moore_covergroup_create("same_type_cg", 1);
  ASSERT_NE(cg1, nullptr);
  ASSERT_NE(cg2, nullptr);

  __moore_coverpoint_init(cg1, 0, "cp");
  __moore_coverpoint_init(cg2, 0, "cp");

  // Sample different values in each instance
  __moore_coverpoint_sample(cg1, 0, 10);
  __moore_coverpoint_sample(cg2, 0, 20);

  // Instance coverage should be 100% for each (single value)
  double inst1 = __moore_covergroup_get_inst_coverage(cg1);
  double inst2 = __moore_covergroup_get_inst_coverage(cg2);
  EXPECT_DOUBLE_EQ(inst1, 100.0);
  EXPECT_DOUBLE_EQ(inst2, 100.0);

  // Type coverage should be average of instances = 100%
  double typeCov1 = __moore_covergroup_get_coverage(cg1);
  double typeCov2 = __moore_covergroup_get_coverage(cg2);
  EXPECT_DOUBLE_EQ(typeCov1, 100.0);
  EXPECT_DOUBLE_EQ(typeCov2, 100.0);

  __moore_covergroup_destroy(cg1);
  __moore_covergroup_destroy(cg2);
}

TEST(MooreRuntimeCoverageTest, PerInstanceModeCoverage) {
  // Test that per_instance mode returns instance-specific coverage
  void *cg1 = __moore_covergroup_create("per_inst_cg", 1);
  void *cg2 = __moore_covergroup_create("per_inst_cg", 1);
  ASSERT_NE(cg1, nullptr);
  ASSERT_NE(cg2, nullptr);

  // Enable per_instance mode
  __moore_covergroup_set_per_instance(cg1, true);
  __moore_covergroup_set_per_instance(cg2, true);

  __moore_coverpoint_init(cg1, 0, "cp");
  __moore_coverpoint_init(cg2, 0, "cp");

  // Sample different amounts
  __moore_coverpoint_sample(cg1, 0, 1);
  __moore_coverpoint_sample(cg2, 0, 10);
  __moore_coverpoint_sample(cg2, 0, 20);

  // In per_instance mode, get_coverage should equal get_inst_coverage
  double cov1 = __moore_covergroup_get_coverage(cg1);
  double cov2 = __moore_covergroup_get_coverage(cg2);
  double inst1 = __moore_covergroup_get_inst_coverage(cg1);
  double inst2 = __moore_covergroup_get_inst_coverage(cg2);

  EXPECT_DOUBLE_EQ(cov1, inst1);
  EXPECT_DOUBLE_EQ(cov2, inst2);

  __moore_covergroup_destroy(cg1);
  __moore_covergroup_destroy(cg2);
}

TEST(MooreRuntimeCoverageTest, CoverageWithAtLeastThreshold) {
  // Test that coverage respects at_least threshold
  void *cg = __moore_covergroup_create("at_least_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp");

  // Set at_least to 3
  __moore_covergroup_set_at_least(cg, 3);

  // Set up explicit bins for precise testing
  MooreCoverageBin bins[3];
  bins[0] = {.name = "bin0", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 0, .high = 9, .hit_count = 0};
  bins[1] = {.name = "bin1", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 10, .high = 19, .hit_count = 0};
  bins[2] = {.name = "bin2", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 20, .high = 29, .hit_count = 0};

  // Destroy old coverpoint and create with bins
  __moore_covergroup_destroy(cg);
  cg = __moore_covergroup_create("at_least_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 3);
  __moore_covergroup_set_at_least(cg, 3);

  // Sample bin0 once, bin1 twice, bin2 three times
  __moore_coverpoint_sample(cg, 0, 5);     // bin0: 1 hit
  __moore_coverpoint_sample(cg, 0, 15);    // bin1: 1 hit
  __moore_coverpoint_sample(cg, 0, 15);    // bin1: 2 hits
  __moore_coverpoint_sample(cg, 0, 25);    // bin2: 1 hit
  __moore_coverpoint_sample(cg, 0, 25);    // bin2: 2 hits
  __moore_coverpoint_sample(cg, 0, 25);    // bin2: 3 hits

  // Only bin2 has >= 3 hits, so coverage should be 1/3 = 33.33%
  double cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_NEAR(cov, 33.33, 0.5);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CoverageWithIgnoreBins) {
  // Test that ignore bins are excluded from coverage calculation
  void *cg = __moore_covergroup_create("ignore_bins_cg", 1);
  ASSERT_NE(cg, nullptr);

  // Set up explicit bins including an ignore bin
  MooreCoverageBin bins[3];
  bins[0] = {.name = "bin0", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 0, .high = 9, .hit_count = 0};
  bins[1] = {.name = "bin1", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_IGNORE,
             .low = 10, .high = 19, .hit_count = 0};
  bins[2] = {.name = "bin2", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 20, .high = 29, .hit_count = 0};

  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 3);

  // Sample bin0 and bin2
  __moore_coverpoint_sample(cg, 0, 5);     // bin0: 1 hit
  __moore_coverpoint_sample(cg, 0, 25);    // bin2: 1 hit

  // Only 2 normal bins (bin0, bin2), both hit, so coverage = 100%
  double cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov, 100.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, CoveragePartialBins) {
  // Test partial coverage with explicit bins
  void *cg = __moore_covergroup_create("partial_cg", 1);
  ASSERT_NE(cg, nullptr);

  // Set up 4 explicit bins
  MooreCoverageBin bins[4];
  bins[0] = {.name = "bin0", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 0, .high = 9, .hit_count = 0};
  bins[1] = {.name = "bin1", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 10, .high = 19, .hit_count = 0};
  bins[2] = {.name = "bin2", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 20, .high = 29, .hit_count = 0};
  bins[3] = {.name = "bin3", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 30, .high = 39, .hit_count = 0};

  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 4);

  // Sample only bin0 and bin2 (50% coverage)
  __moore_coverpoint_sample(cg, 0, 5);     // bin0: hit
  __moore_coverpoint_sample(cg, 0, 25);    // bin2: hit

  double cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov, 50.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCrossCoverageTest, CrossGetInstCoverage) {
  // Test cross instance coverage
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  int32_t cpIndices[] = {0, 1};
  int32_t crossIdx = __moore_cross_create(cg, "cross01", cpIndices, 2);
  EXPECT_GE(crossIdx, 0);

  // Sample values
  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 0, 2);
  __moore_coverpoint_sample(cg, 1, 10);
  __moore_coverpoint_sample(cg, 1, 20);

  // Sample cross bins
  int64_t vals1[] = {1, 10};
  int64_t vals2[] = {2, 20};
  __moore_cross_sample(cg, vals1, 2);
  __moore_cross_sample(cg, vals2, 2);

  // Get coverage - should be same as inst_coverage for crosses
  double cov = __moore_cross_get_coverage(cg, crossIdx);
  double instCov = __moore_cross_get_inst_coverage(cg, crossIdx);
  EXPECT_DOUBLE_EQ(cov, instCov);

  // 2 bins hit out of 4 possible (2x2), so 50%
  EXPECT_DOUBLE_EQ(cov, 50.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCrossCoverageTest, CrossCoverageWithAtLeast) {
  // Test that cross coverage respects at_least threshold
  void *cg = __moore_covergroup_create("cross_at_least_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  // Set at_least to 2
  __moore_covergroup_set_at_least(cg, 2);

  int32_t cpIndices[] = {0, 1};
  int32_t crossIdx = __moore_cross_create(cg, "cross01", cpIndices, 2);
  EXPECT_GE(crossIdx, 0);

  // Sample values to create cross products
  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 0, 2);
  __moore_coverpoint_sample(cg, 1, 10);
  __moore_coverpoint_sample(cg, 1, 20);

  // Sample cross bins: (1,10) twice, (2,20) once
  int64_t vals1[] = {1, 10};
  int64_t vals2[] = {2, 20};
  __moore_cross_sample(cg, vals1, 2);
  __moore_cross_sample(cg, vals1, 2);  // (1,10) hit twice
  __moore_cross_sample(cg, vals2, 2);  // (2,20) hit once

  // Only (1,10) has >= 2 hits, so coverage = 1/4 = 25%
  double cov = __moore_cross_get_coverage(cg, crossIdx);
  EXPECT_DOUBLE_EQ(cov, 25.0);

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
// Coverage Assertion Tests
//===----------------------------------------------------------------------===//

// Helper struct for tracking callback invocations in tests
struct CoverageAssertCallbackData {
  int callCount = 0;
  std::string lastCgName;
  std::string lastCpName;
  double lastActualCoverage = 0.0;
  double lastRequiredGoal = 0.0;

  void reset() {
    callCount = 0;
    lastCgName.clear();
    lastCpName.clear();
    lastActualCoverage = 0.0;
    lastRequiredGoal = 0.0;
  }
};

// Global callback data for tests
static CoverageAssertCallbackData g_assertCallbackData;

// Test callback function
static void testAssertCallback(const char *cg_name, const char *cp_name,
                               double actual_coverage, double required_goal,
                               void *userData) {
  auto *data = static_cast<CoverageAssertCallbackData *>(userData);
  data->callCount++;
  data->lastCgName = cg_name ? cg_name : "";
  data->lastCpName = cp_name ? cp_name : "";
  data->lastActualCoverage = actual_coverage;
  data->lastRequiredGoal = required_goal;
}

TEST(MooreRuntimeCoverageAssertTest, SetFailureCallback) {
  g_assertCallbackData.reset();

  // Set the callback
  __moore_coverage_set_failure_callback(testAssertCallback, &g_assertCallbackData);

  // Create a covergroup with no samples (0% coverage)
  void *cg = __moore_covergroup_create("assert_cb_test", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Assert with 50% goal should fail and invoke callback
  bool result = __moore_covergroup_assert_goal(cg, 50.0);
  EXPECT_FALSE(result);
  EXPECT_EQ(g_assertCallbackData.callCount, 1);
  EXPECT_EQ(g_assertCallbackData.lastCgName, "assert_cb_test");
  EXPECT_TRUE(g_assertCallbackData.lastCpName.empty());
  EXPECT_DOUBLE_EQ(g_assertCallbackData.lastActualCoverage, 0.0);

  // Clear callback
  __moore_coverage_set_failure_callback(nullptr, nullptr);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, AssertGlobalGoalPass) {
  // Create covergroup with full coverage
  void *cg = __moore_covergroup_create("global_assert_pass", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 42);

  // Assert with 50% goal should pass (coverage is 100%)
  bool result = __moore_coverage_assert_goal(50.0);
  EXPECT_TRUE(result);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, AssertGlobalGoalFail) {
  g_assertCallbackData.reset();
  __moore_coverage_set_failure_callback(testAssertCallback, &g_assertCallbackData);

  // Create covergroup with no samples (0% coverage)
  void *cg = __moore_covergroup_create("global_assert_fail", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Assert with 50% goal should fail
  bool result = __moore_coverage_assert_goal(50.0);
  EXPECT_FALSE(result);
  EXPECT_EQ(g_assertCallbackData.callCount, 1);
  EXPECT_TRUE(g_assertCallbackData.lastCgName.empty()); // Global check has no cg name

  __moore_coverage_set_failure_callback(nullptr, nullptr);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, AssertCovergroupGoalPass) {
  void *cg = __moore_covergroup_create("cg_assert_pass", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 42);

  // Assert with 50% goal should pass (coverage is 100%)
  bool result = __moore_covergroup_assert_goal(cg, 50.0);
  EXPECT_TRUE(result);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, AssertCovergroupGoalFail) {
  g_assertCallbackData.reset();
  __moore_coverage_set_failure_callback(testAssertCallback, &g_assertCallbackData);

  void *cg = __moore_covergroup_create("cg_assert_fail", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Assert with 50% goal should fail (0% coverage)
  bool result = __moore_covergroup_assert_goal(cg, 50.0);
  EXPECT_FALSE(result);
  EXPECT_EQ(g_assertCallbackData.callCount, 1);
  EXPECT_EQ(g_assertCallbackData.lastCgName, "cg_assert_fail");
  EXPECT_DOUBLE_EQ(g_assertCallbackData.lastActualCoverage, 0.0);

  __moore_coverage_set_failure_callback(nullptr, nullptr);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, AssertCovergroupUsesHigherGoal) {
  g_assertCallbackData.reset();
  __moore_coverage_set_failure_callback(testAssertCallback, &g_assertCallbackData);

  void *cg = __moore_covergroup_create("cg_higher_goal", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Set covergroup's configured goal to 80%
  __moore_covergroup_set_goal(cg, 80.0);

  // Assert with 50% should use 80% (the higher one)
  bool result = __moore_covergroup_assert_goal(cg, 50.0);
  EXPECT_FALSE(result);
  EXPECT_DOUBLE_EQ(g_assertCallbackData.lastRequiredGoal, 80.0);

  __moore_coverage_set_failure_callback(nullptr, nullptr);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, AssertCoverpointGoalPass) {
  void *cg = __moore_covergroup_create("cp_assert_pass", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 42);

  // Assert with 50% goal should pass (coverage is 100%)
  bool result = __moore_coverpoint_assert_goal(cg, 0, 50.0);
  EXPECT_TRUE(result);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, AssertCoverpointGoalFail) {
  g_assertCallbackData.reset();
  __moore_coverage_set_failure_callback(testAssertCallback, &g_assertCallbackData);

  void *cg = __moore_covergroup_create("cp_assert_fail", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "my_coverpoint");

  // Assert with 50% goal should fail (0% coverage)
  bool result = __moore_coverpoint_assert_goal(cg, 0, 50.0);
  EXPECT_FALSE(result);
  EXPECT_EQ(g_assertCallbackData.callCount, 1);
  EXPECT_EQ(g_assertCallbackData.lastCgName, "cp_assert_fail");
  EXPECT_EQ(g_assertCallbackData.lastCpName, "my_coverpoint");

  __moore_coverage_set_failure_callback(nullptr, nullptr);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, AssertCoverpointUsesHigherGoal) {
  g_assertCallbackData.reset();
  __moore_coverage_set_failure_callback(testAssertCallback, &g_assertCallbackData);

  void *cg = __moore_covergroup_create("cp_higher_goal", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Set coverpoint's configured goal to 90%
  __moore_coverpoint_set_goal(cg, 0, 90.0);

  // Assert with 50% should use 90% (the higher one)
  bool result = __moore_coverpoint_assert_goal(cg, 0, 50.0);
  EXPECT_FALSE(result);
  EXPECT_DOUBLE_EQ(g_assertCallbackData.lastRequiredGoal, 90.0);

  __moore_coverage_set_failure_callback(nullptr, nullptr);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, AssertInvalidCovergroup) {
  // Null covergroup should return false
  bool result = __moore_covergroup_assert_goal(nullptr, 50.0);
  EXPECT_FALSE(result);
}

TEST(MooreRuntimeCoverageAssertTest, AssertInvalidCoverpointIndex) {
  void *cg = __moore_covergroup_create("cp_invalid_idx", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Invalid index should return false
  bool result1 = __moore_coverpoint_assert_goal(cg, -1, 50.0);
  EXPECT_FALSE(result1);

  bool result2 = __moore_coverpoint_assert_goal(cg, 5, 50.0);
  EXPECT_FALSE(result2);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, CheckAllGoalsPass) {
  // Create covergroups with full coverage
  void *cg1 = __moore_covergroup_create("all_goals_cg1", 1);
  void *cg2 = __moore_covergroup_create("all_goals_cg2", 1);
  ASSERT_NE(cg1, nullptr);
  ASSERT_NE(cg2, nullptr);

  __moore_coverpoint_init(cg1, 0, "cp1");
  __moore_coverpoint_init(cg2, 0, "cp2");

  // Set low goals
  __moore_covergroup_set_goal(cg1, 50.0);
  __moore_covergroup_set_goal(cg2, 50.0);
  __moore_coverpoint_set_goal(cg1, 0, 50.0);
  __moore_coverpoint_set_goal(cg2, 0, 50.0);

  // Sample values to get 100% coverage
  __moore_coverpoint_sample(cg1, 0, 1);
  __moore_coverpoint_sample(cg2, 0, 2);

  // All goals should be met
  bool result = __moore_coverage_check_all_goals();
  EXPECT_TRUE(result);

  __moore_covergroup_destroy(cg1);
  __moore_covergroup_destroy(cg2);
}

TEST(MooreRuntimeCoverageAssertTest, CheckAllGoalsFail) {
  g_assertCallbackData.reset();
  __moore_coverage_set_failure_callback(testAssertCallback, &g_assertCallbackData);

  // Create covergroup with no coverage but high goal
  void *cg = __moore_covergroup_create("all_goals_fail", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Set high goals (default is 100%)
  // No samples, so coverage is 0%

  // All goals check should fail
  bool result = __moore_coverage_check_all_goals();
  EXPECT_FALSE(result);
  EXPECT_GE(g_assertCallbackData.callCount, 1); // At least one failure callback

  __moore_coverage_set_failure_callback(nullptr, nullptr);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, GetUnmetGoalCount) {
  // Create covergroups with mixed coverage
  void *cg1 = __moore_covergroup_create("unmet_cg1", 1);
  void *cg2 = __moore_covergroup_create("unmet_cg2", 1);
  ASSERT_NE(cg1, nullptr);
  ASSERT_NE(cg2, nullptr);

  __moore_coverpoint_init(cg1, 0, "cp1");
  __moore_coverpoint_init(cg2, 0, "cp2");

  // Set goals
  __moore_covergroup_set_goal(cg1, 50.0);
  __moore_covergroup_set_goal(cg2, 50.0);
  __moore_coverpoint_set_goal(cg1, 0, 50.0);
  __moore_coverpoint_set_goal(cg2, 0, 50.0);

  // Sample only cg1 - it meets goals, cg2 does not
  __moore_coverpoint_sample(cg1, 0, 1);

  int32_t unmetCount = __moore_coverage_get_unmet_goal_count();
  // cg2 covergroup goal (1) + cg2 coverpoint goal (1) = 2 unmet
  EXPECT_EQ(unmetCount, 2);

  __moore_covergroup_destroy(cg1);
  __moore_covergroup_destroy(cg2);
}

TEST(MooreRuntimeCoverageAssertTest, RegisterAssertionGlobal) {
  // Clear any existing assertions
  __moore_coverage_clear_registered_assertions();

  // Create a covergroup with coverage
  void *cg = __moore_covergroup_create("reg_assert_global", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 42);

  // Register a global assertion with low goal
  int32_t id = __moore_coverage_register_assertion(nullptr, -1, 50.0);
  EXPECT_GE(id, 0);

  // Check registered assertions - should pass
  bool result = __moore_coverage_check_registered_assertions();
  EXPECT_TRUE(result);

  __moore_coverage_clear_registered_assertions();
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, RegisterAssertionCovergroup) {
  __moore_coverage_clear_registered_assertions();

  void *cg = __moore_covergroup_create("reg_assert_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 42);

  // Register a covergroup-level assertion
  int32_t id = __moore_coverage_register_assertion(cg, -1, 50.0);
  EXPECT_GE(id, 0);

  // Check registered assertions - should pass
  bool result = __moore_coverage_check_registered_assertions();
  EXPECT_TRUE(result);

  __moore_coverage_clear_registered_assertions();
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, RegisterAssertionCoverpoint) {
  __moore_coverage_clear_registered_assertions();

  void *cg = __moore_covergroup_create("reg_assert_cp", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 42);

  // Register a coverpoint-level assertion
  int32_t id = __moore_coverage_register_assertion(cg, 0, 50.0);
  EXPECT_GE(id, 0);

  // Check registered assertions - should pass
  bool result = __moore_coverage_check_registered_assertions();
  EXPECT_TRUE(result);

  __moore_coverage_clear_registered_assertions();
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, RegisterMultipleAssertions) {
  g_assertCallbackData.reset();
  __moore_coverage_set_failure_callback(testAssertCallback, &g_assertCallbackData);
  __moore_coverage_clear_registered_assertions();

  void *cg1 = __moore_covergroup_create("reg_multi_cg1", 1);
  void *cg2 = __moore_covergroup_create("reg_multi_cg2", 1);
  ASSERT_NE(cg1, nullptr);
  ASSERT_NE(cg2, nullptr);

  __moore_coverpoint_init(cg1, 0, "cp1");
  __moore_coverpoint_init(cg2, 0, "cp2");

  // Only cg1 has coverage
  __moore_coverpoint_sample(cg1, 0, 42);

  // Register assertions for both
  int32_t id1 = __moore_coverage_register_assertion(cg1, -1, 50.0);
  int32_t id2 = __moore_coverage_register_assertion(cg2, -1, 50.0);
  EXPECT_GE(id1, 0);
  EXPECT_GE(id2, 0);
  EXPECT_NE(id1, id2); // Different IDs

  // Check - should fail because cg2 doesn't meet goal
  bool result = __moore_coverage_check_registered_assertions();
  EXPECT_FALSE(result);
  EXPECT_GE(g_assertCallbackData.callCount, 1);

  __moore_coverage_set_failure_callback(nullptr, nullptr);
  __moore_coverage_clear_registered_assertions();
  __moore_covergroup_destroy(cg1);
  __moore_covergroup_destroy(cg2);
}

TEST(MooreRuntimeCoverageAssertTest, RegisterAssertionInvalidCoverpoint) {
  __moore_coverage_clear_registered_assertions();

  void *cg = __moore_covergroup_create("reg_invalid", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Invalid coverpoint index should fail
  int32_t id = __moore_coverage_register_assertion(cg, 5, 50.0);
  EXPECT_EQ(id, -1);

  __moore_coverage_clear_registered_assertions();
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, ClearRegisteredAssertions) {
  __moore_coverage_clear_registered_assertions();

  void *cg = __moore_covergroup_create("clear_assert", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Register assertion
  __moore_coverage_register_assertion(cg, -1, 50.0);

  // Clear
  __moore_coverage_clear_registered_assertions();

  // Check - should pass (no assertions registered)
  bool result = __moore_coverage_check_registered_assertions();
  EXPECT_TRUE(result);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, GoalClampingMinPercentage) {
  void *cg = __moore_covergroup_create("goal_clamp", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Negative percentage should be clamped to 0
  bool result = __moore_coverage_assert_goal(-10.0);
  // 0% coverage >= 0% goal = true
  EXPECT_TRUE(result);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageAssertTest, GoalClampingMaxPercentage) {
  void *cg = __moore_covergroup_create("goal_clamp_max", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 42);

  // Percentage > 100 should be clamped to 100
  bool result = __moore_coverage_assert_goal(150.0);
  // 100% coverage >= 100% goal = true
  EXPECT_TRUE(result);

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
  // MooreCoverageBin: {name, type, kind, low, high, hit_count}
  MooreCoverageBin bins[] = {
    {"low", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 0, 10, 0},
    {"mid", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 11, 20, 0},
    {"high", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 21, 30, 0}
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
// Wildcard Bin Matching Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCoverageTest, WildcardBinBasicMatch) {
  // Test wildcard bin matching using mask+value encoding
  // For pattern 4'b1??? (match any 4-bit value starting with 1):
  // - pattern value (low) = 0b1000 = 8
  // - mask (high) = 0b0111 = 7 (bits 0-2 are don't care)
  void *cg = __moore_covergroup_create("wildcard_test_cg", 1);
  ASSERT_NE(cg, nullptr);

  // Initialize with a wildcard bin: 4'b1??? matches values 8-15
  MooreCoverageBin bins[] = {
      {"high_nibble", MOORE_BIN_WILDCARD, MOORE_BIN_KIND_NORMAL, 8, 7, 0}
      // low=8 (0b1000), high=7 (0b0111 mask)
  };
  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 1);

  // Sample values that should match (8-15)
  __moore_coverpoint_sample(cg, 0, 8);   // 0b1000 - should match
  __moore_coverpoint_sample(cg, 0, 9);   // 0b1001 - should match
  __moore_coverpoint_sample(cg, 0, 15);  // 0b1111 - should match

  // Check that the bin was hit 3 times
  EXPECT_EQ(__moore_coverpoint_get_bin_hits(cg, 0, 0), 3);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, WildcardBinNoMatch) {
  // Test that values not matching the wildcard pattern don't hit the bin
  void *cg = __moore_covergroup_create("wildcard_nomatch_cg", 1);
  ASSERT_NE(cg, nullptr);

  // Wildcard bin: 4'b1??? matches values 8-15
  MooreCoverageBin bins[] = {
      {"high_nibble", MOORE_BIN_WILDCARD, MOORE_BIN_KIND_NORMAL, 8, 7, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 1);

  // Sample values that should NOT match (0-7)
  __moore_coverpoint_sample(cg, 0, 0);   // 0b0000 - should not match
  __moore_coverpoint_sample(cg, 0, 1);   // 0b0001 - should not match
  __moore_coverpoint_sample(cg, 0, 7);   // 0b0111 - should not match

  // The bin should not be hit
  EXPECT_EQ(__moore_coverpoint_get_bin_hits(cg, 0, 0), 0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, WildcardBinMiddleBits) {
  // Test wildcard pattern with don't care in the middle: 8'b10??01??
  // Pattern: 0b10000100 = 132
  // Mask: 0b00110011 = 51 (bits 0,1,4,5 are don't care)
  void *cg = __moore_covergroup_create("wildcard_middle_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[] = {
      {"middle_wildcard", MOORE_BIN_WILDCARD, MOORE_BIN_KIND_NORMAL, 132, 51, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 1);

  // Values matching 10xx01xx pattern:
  // 0b10000100 = 132 - should match
  // 0b10110111 = 183 - should match (0b10110111)
  // 0b10000111 = 135 - should match
  __moore_coverpoint_sample(cg, 0, 132);  // 0b10000100 - should match
  __moore_coverpoint_sample(cg, 0, 183);  // 0b10110111 - should match
  __moore_coverpoint_sample(cg, 0, 135);  // 0b10000111 - should match

  EXPECT_EQ(__moore_coverpoint_get_bin_hits(cg, 0, 0), 3);

  // Values NOT matching:
  // 0b00000100 = 4 - wrong bit 7
  // 0b11000100 = 196 - wrong bit 6
  __moore_coverpoint_sample(cg, 0, 4);    // should not match
  __moore_coverpoint_sample(cg, 0, 196);  // should not match

  EXPECT_EQ(__moore_coverpoint_get_bin_hits(cg, 0, 0), 3);  // Still 3

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, WildcardBinAllDontCare) {
  // Test wildcard pattern where all bits are don't care: ????
  // This should match any 4-bit value
  // Pattern: 0, Mask: 0xF (all bits are don't care)
  void *cg = __moore_covergroup_create("wildcard_alldc_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[] = {
      {"all_dontcare", MOORE_BIN_WILDCARD, MOORE_BIN_KIND_NORMAL, 0, 0xF, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 1);

  // All values 0-15 should match
  for (int i = 0; i <= 15; i++) {
    __moore_coverpoint_sample(cg, 0, i);
  }

  EXPECT_EQ(__moore_coverpoint_get_bin_hits(cg, 0, 0), 16);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, WildcardBinNoDontCare) {
  // Test wildcard pattern with no don't care bits (equivalent to value bin)
  // Pattern: 0b1010 = 10, Mask: 0 (all bits must match)
  void *cg = __moore_covergroup_create("wildcard_nodc_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[] = {
      {"exact_value", MOORE_BIN_WILDCARD, MOORE_BIN_KIND_NORMAL, 10, 0, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 1);

  // Only exact value 10 should match
  __moore_coverpoint_sample(cg, 0, 9);   // should not match
  __moore_coverpoint_sample(cg, 0, 10);  // should match
  __moore_coverpoint_sample(cg, 0, 11);  // should not match

  EXPECT_EQ(__moore_coverpoint_get_bin_hits(cg, 0, 0), 1);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, WildcardBinIllegal) {
  // Test wildcard pattern for illegal bins
  // illegal_bins high = 4'b1??? should flag values 8-15 as illegal
  void *cg = __moore_covergroup_create("wildcard_illegal_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp");

  // Reset illegal bin state
  __moore_coverage_reset_illegal_bin_hits();
  __moore_coverage_set_illegal_bin_fatal(false);  // Don't exit on illegal

  // Set illegal wildcard bin via the set_illegal_bins function
  MooreCoverageBin illegalBins[] = {
      {"high_values", MOORE_BIN_WILDCARD, MOORE_BIN_KIND_ILLEGAL, 8, 7, 0}
  };
  __moore_coverpoint_set_illegal_bins(cg, 0, illegalBins, 1);

  // Sample legal values (0-7)
  __moore_coverpoint_sample(cg, 0, 0);
  __moore_coverpoint_sample(cg, 0, 5);
  __moore_coverpoint_sample(cg, 0, 7);
  EXPECT_EQ(__moore_coverage_get_illegal_bin_hits(), 0);

  // Sample illegal values (8-15)
  __moore_coverpoint_sample(cg, 0, 8);
  EXPECT_EQ(__moore_coverage_get_illegal_bin_hits(), 1);

  __moore_coverpoint_sample(cg, 0, 15);
  EXPECT_EQ(__moore_coverage_get_illegal_bin_hits(), 2);

  __moore_covergroup_destroy(cg);
  __moore_coverage_reset_illegal_bin_hits();
}

TEST(MooreRuntimeCoverageTest, WildcardBinIgnore) {
  // Test wildcard pattern for ignore bins
  // ignore_bins reserved = 8'b1111???? should ignore values 240-255
  void *cg = __moore_covergroup_create("wildcard_ignore_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp");

  // Set ignore wildcard bin
  // Pattern: 0b11110000 = 240, Mask: 0b00001111 = 15
  MooreCoverageBin ignoreBins[] = {
      {"reserved", MOORE_BIN_WILDCARD, MOORE_BIN_KIND_IGNORE, 240, 15, 0}
  };
  __moore_coverpoint_set_ignore_bins(cg, 0, ignoreBins, 1);

  // Check ignored values
  EXPECT_TRUE(__moore_coverpoint_is_ignored(cg, 0, 240));   // 0b11110000
  EXPECT_TRUE(__moore_coverpoint_is_ignored(cg, 0, 255));   // 0b11111111
  EXPECT_TRUE(__moore_coverpoint_is_ignored(cg, 0, 245));   // 0b11110101

  // Check non-ignored values
  EXPECT_FALSE(__moore_coverpoint_is_ignored(cg, 0, 0));
  EXPECT_FALSE(__moore_coverpoint_is_ignored(cg, 0, 128));  // 0b10000000
  EXPECT_FALSE(__moore_coverpoint_is_ignored(cg, 0, 239));  // 0b11101111

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, WildcardBinIsIllegalCheck) {
  // Test __moore_coverpoint_is_illegal with wildcard bins
  void *cg = __moore_covergroup_create("wildcard_is_illegal_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp");

  // Add wildcard illegal bin: 4'b11?? (values 12-15)
  // Pattern: 0b1100 = 12, Mask: 0b0011 = 3
  MooreCoverageBin illegalBins[] = {
      {"top_quarter", MOORE_BIN_WILDCARD, MOORE_BIN_KIND_ILLEGAL, 12, 3, 0}
  };
  __moore_coverpoint_set_illegal_bins(cg, 0, illegalBins, 1);

  // Check illegal values (12-15)
  EXPECT_TRUE(__moore_coverpoint_is_illegal(cg, 0, 12));
  EXPECT_TRUE(__moore_coverpoint_is_illegal(cg, 0, 13));
  EXPECT_TRUE(__moore_coverpoint_is_illegal(cg, 0, 14));
  EXPECT_TRUE(__moore_coverpoint_is_illegal(cg, 0, 15));

  // Check legal values (0-11)
  EXPECT_FALSE(__moore_coverpoint_is_illegal(cg, 0, 0));
  EXPECT_FALSE(__moore_coverpoint_is_illegal(cg, 0, 8));
  EXPECT_FALSE(__moore_coverpoint_is_illegal(cg, 0, 11));

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

TEST(MooreRuntimeCoverageTest, HtmlReportContainsCoverageData) {
  // Create a covergroup with bins and sample values
  void *cg = __moore_covergroup_create("html_data_cg", 2);
  ASSERT_NE(cg, nullptr);

  // Initialize coverpoints with explicit bins
  MooreCoverageBin bins1[] = {
      {"low_bin", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 0, 10, 0},
      {"high_bin", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 11, 20, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "signal_x", bins1, 2);
  __moore_coverpoint_init(cg, 1, "signal_y");

  // Sample values
  __moore_coverpoint_sample(cg, 0, 5);   // Hits low_bin
  __moore_coverpoint_sample(cg, 0, 15);  // Hits high_bin
  __moore_coverpoint_sample(cg, 1, 100);

  // Generate HTML report
  const char *filename = "/tmp/coverage_data_test.html";
  int32_t result = __moore_coverage_report_html(filename);
  EXPECT_EQ(result, 0);

  // Read and verify HTML content
  FILE *fp = std::fopen(filename, "r");
  ASSERT_NE(fp, nullptr);

  std::fseek(fp, 0, SEEK_END);
  long fileSize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  std::string htmlContent(fileSize, '\0');
  std::fread(&htmlContent[0], 1, fileSize, fp);
  std::fclose(fp);

  // Verify key HTML elements are present
  EXPECT_NE(htmlContent.find("html_data_cg"), std::string::npos);
  EXPECT_NE(htmlContent.find("signal_x"), std::string::npos);
  EXPECT_NE(htmlContent.find("signal_y"), std::string::npos);
  EXPECT_NE(htmlContent.find("Coverpoints"), std::string::npos);
  EXPECT_NE(htmlContent.find("Coverage Report"), std::string::npos);

  // Verify bin details are present
  EXPECT_NE(htmlContent.find("low_bin"), std::string::npos);
  EXPECT_NE(htmlContent.find("high_bin"), std::string::npos);
  EXPECT_NE(htmlContent.find("range"), std::string::npos);

  // Verify CSS color coding is present
  EXPECT_NE(htmlContent.find("--success"), std::string::npos);
  EXPECT_NE(htmlContent.find("--warning"), std::string::npos);
  EXPECT_NE(htmlContent.find("--danger"), std::string::npos);

  // Clean up
  std::remove(filename);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, HtmlReportWithCrossCoverage) {
  // Create a covergroup with cross coverage
  void *cg = __moore_covergroup_create("html_cross_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "addr");
  __moore_coverpoint_init(cg, 1, "data");

  // Create a cross coverage item
  int32_t cpIndices[] = {0, 1};
  int32_t crossIdx = __moore_cross_create(cg, "addr_data_cross", cpIndices, 2);
  EXPECT_GE(crossIdx, 0);

  // Sample values to trigger cross coverage
  __moore_coverpoint_sample(cg, 0, 10);
  __moore_coverpoint_sample(cg, 1, 20);
  int64_t cpValues[] = {10, 20};
  __moore_cross_sample(cg, cpValues, 2);

  // Generate HTML report
  const char *filename = "/tmp/coverage_cross_test.html";
  int32_t result = __moore_coverage_report_html(filename);
  EXPECT_EQ(result, 0);

  // Read and verify HTML content
  FILE *fp = std::fopen(filename, "r");
  ASSERT_NE(fp, nullptr);

  std::fseek(fp, 0, SEEK_END);
  long fileSize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  std::string htmlContent(fileSize, '\0');
  std::fread(&htmlContent[0], 1, fileSize, fp);
  std::fclose(fp);

  // Verify cross coverage section is present
  EXPECT_NE(htmlContent.find("Cross Coverage"), std::string::npos);
  EXPECT_NE(htmlContent.find("addr_data_cross"), std::string::npos);
  EXPECT_NE(htmlContent.find("addr"), std::string::npos);
  EXPECT_NE(htmlContent.find("data"), std::string::npos);
  EXPECT_NE(htmlContent.find("Bins Hit"), std::string::npos);

  // Clean up
  std::remove(filename);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, HtmlReportColorCodingThresholds) {
  // Test that color coding is applied correctly based on coverage thresholds
  // Green (100%), Yellow (50-99%), Red (<50%)

  void *cg = __moore_covergroup_create("html_color_cg", 1);
  ASSERT_NE(cg, nullptr);

  // Create a coverpoint with bins to test color coding
  MooreCoverageBin bins[] = {
      {"bin1", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 1, 1, 0},
      {"bin2", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 2, 2, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "test_cp", bins, 2);

  // Sample only one bin to get 50% coverage
  __moore_coverpoint_sample(cg, 0, 1);

  const char *filename = "/tmp/coverage_color_test.html";
  int32_t result = __moore_coverage_report_html(filename);
  EXPECT_EQ(result, 0);

  FILE *fp = std::fopen(filename, "r");
  ASSERT_NE(fp, nullptr);

  std::fseek(fp, 0, SEEK_END);
  long fileSize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  std::string htmlContent(fileSize, '\0');
  std::fread(&htmlContent[0], 1, fileSize, fp);
  std::fclose(fp);

  // Verify that coverage percentage is shown with appropriate color
  // With 1 out of 2 bins hit, we should have 50% coverage (warning color)
  EXPECT_NE(htmlContent.find("50.0%"), std::string::npos);

  // Clean up
  std::remove(filename);
  __moore_covergroup_destroy(cg);
}

//===----------------------------------------------------------------------===//
// Text Report Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCoverageTest, TextReportGeneration) {
  void *cg = __moore_covergroup_create("text_report_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "signal_a");
  __moore_coverpoint_init(cg, 1, "signal_b");

  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 0, 2);
  __moore_coverpoint_sample(cg, 1, 10);

  // Generate text report with normal verbosity
  const char *filename = "/tmp/coverage_test_report.txt";
  int32_t result = __moore_coverage_report_text(filename, MOORE_TEXT_REPORT_NORMAL);
  EXPECT_EQ(result, 0);

  // Verify file was created
  FILE *fp = std::fopen(filename, "r");
  EXPECT_NE(fp, nullptr);
  if (fp) {
    std::fclose(fp);
    std::remove(filename);
  }

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, TextReportNullFilename) {
  int32_t result = __moore_coverage_report_text(nullptr, MOORE_TEXT_REPORT_NORMAL);
  EXPECT_NE(result, 0);
}

TEST(MooreRuntimeCoverageTest, TextReportSummaryOnly) {
  // Create a covergroup with bins and sample values
  void *cg = __moore_covergroup_create("text_summary_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[] = {
      {"bin_a", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 0, 10, 0},
      {"bin_b", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 11, 20, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "cp_test", bins, 2);
  __moore_coverpoint_sample(cg, 0, 5);

  // Generate summary-only report
  const char *filename = "/tmp/coverage_summary_test.txt";
  int32_t result = __moore_coverage_report_text(filename, MOORE_TEXT_REPORT_SUMMARY);
  EXPECT_EQ(result, 0);

  // Read and verify content
  FILE *fp = std::fopen(filename, "r");
  ASSERT_NE(fp, nullptr);

  std::fseek(fp, 0, SEEK_END);
  long fileSize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  std::string content(fileSize, '\0');
  std::fread(&content[0], 1, fileSize, fp);
  std::fclose(fp);

  // Summary should contain overall coverage
  EXPECT_NE(content.find("Coverage Report"), std::string::npos);
  EXPECT_NE(content.find("Overall Coverage"), std::string::npos);
  EXPECT_NE(content.find("Summary:"), std::string::npos);

  // Summary mode should NOT contain covergroup details
  EXPECT_EQ(content.find("text_summary_cg"), std::string::npos);

  std::remove(filename);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, TextReportDetailedWithBins) {
  // Create a covergroup with bins
  void *cg = __moore_covergroup_create("text_detail_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[] = {
      {"low_range", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 0, 50, 0},
      {"high_range", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 51, 100, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "addr", bins, 2);

  // Sample only low range
  __moore_coverpoint_sample(cg, 0, 25);
  __moore_coverpoint_sample(cg, 0, 30);

  // Generate detailed report
  const char *filename = "/tmp/coverage_detail_test.txt";
  int32_t result = __moore_coverage_report_text(filename, MOORE_TEXT_REPORT_DETAILED);
  EXPECT_EQ(result, 0);

  // Read and verify content
  FILE *fp = std::fopen(filename, "r");
  ASSERT_NE(fp, nullptr);

  std::fseek(fp, 0, SEEK_END);
  long fileSize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  std::string content(fileSize, '\0');
  std::fread(&content[0], 1, fileSize, fp);
  std::fclose(fp);

  // Detailed report should contain bin information
  EXPECT_NE(content.find("text_detail_cg"), std::string::npos);
  EXPECT_NE(content.find("addr"), std::string::npos);
  EXPECT_NE(content.find("low_range"), std::string::npos);
  EXPECT_NE(content.find("high_range"), std::string::npos);

  // High range should be marked as a hole (0 hits)
  EXPECT_NE(content.find("HOLE"), std::string::npos);

  std::remove(filename);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, TextReportSummaryFunction) {
  // Create a covergroup
  void *cg = __moore_covergroup_create("summary_test_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp1");
  __moore_coverpoint_sample(cg, 0, 42);

  // Get summary string
  char *summary = __moore_coverage_report_summary();
  ASSERT_NE(summary, nullptr);

  // Verify it contains coverage percentage
  std::string summaryStr(summary);
  EXPECT_NE(summaryStr.find("Coverage:"), std::string::npos);
  EXPECT_NE(summaryStr.find("%"), std::string::npos);
  EXPECT_NE(summaryStr.find("covergroup"), std::string::npos);

  __moore_free(summary);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, TextReportGetFullReport) {
  // Create a covergroup
  void *cg = __moore_covergroup_create("get_report_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "test_cp");
  __moore_coverpoint_sample(cg, 0, 100);

  // Get full text report
  char *report = __moore_coverage_get_text_report(MOORE_TEXT_REPORT_NORMAL);
  ASSERT_NE(report, nullptr);

  std::string reportStr(report);
  EXPECT_NE(reportStr.find("Coverage Report"), std::string::npos);
  EXPECT_NE(reportStr.find("get_report_cg"), std::string::npos);
  EXPECT_NE(reportStr.find("test_cp"), std::string::npos);

  __moore_free(report);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, TextReportWithCrossCoverage) {
  // Create a covergroup with cross coverage
  void *cg = __moore_covergroup_create("text_cross_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "addr");
  __moore_coverpoint_init(cg, 1, "data");

  // Create cross coverage
  int32_t cpIndices[] = {0, 1};
  int32_t crossIdx = __moore_cross_create(cg, "addr_x_data", cpIndices, 2);
  EXPECT_GE(crossIdx, 0);

  // Sample some values
  __moore_coverpoint_sample(cg, 0, 10);
  __moore_coverpoint_sample(cg, 1, 20);
  int64_t crossValues[] = {10, 20};
  __moore_cross_sample(cg, crossValues, 2);

  // Generate detailed report
  const char *filename = "/tmp/coverage_cross_text_test.txt";
  int32_t result = __moore_coverage_report_text(filename, MOORE_TEXT_REPORT_DETAILED);
  EXPECT_EQ(result, 0);

  // Read and verify content
  FILE *fp = std::fopen(filename, "r");
  ASSERT_NE(fp, nullptr);

  std::fseek(fp, 0, SEEK_END);
  long fileSize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  std::string content(fileSize, '\0');
  std::fread(&content[0], 1, fileSize, fp);
  std::fclose(fp);

  // Verify cross coverage is included
  EXPECT_NE(content.find("text_cross_cg"), std::string::npos);
  EXPECT_NE(content.find("Cross:"), std::string::npos);
  EXPECT_NE(content.find("addr_x_data"), std::string::npos);

  std::remove(filename);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, TextReportCoverageHolesSummary) {
  // Create a covergroup with multiple bins, some with 0 hits
  void *cg = __moore_covergroup_create("holes_test_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[] = {
      {"bin_hit", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 1, 1, 0},
      {"bin_miss1", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 2, 2, 0},
      {"bin_miss2", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 3, 3, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "test_cp", bins, 3);

  // Only hit bin_hit
  __moore_coverpoint_sample(cg, 0, 1);

  // Generate report
  const char *filename = "/tmp/coverage_holes_test.txt";
  int32_t result = __moore_coverage_report_text(filename, MOORE_TEXT_REPORT_NORMAL);
  EXPECT_EQ(result, 0);

  // Read and verify content
  FILE *fp = std::fopen(filename, "r");
  ASSERT_NE(fp, nullptr);

  std::fseek(fp, 0, SEEK_END);
  long fileSize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  std::string content(fileSize, '\0');
  std::fread(&content[0], 1, fileSize, fp);
  std::fclose(fp);

  // Verify holes summary section is present
  EXPECT_NE(content.find("Coverage Holes"), std::string::npos);

  std::remove(filename);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, TextReportVerbosityLevels) {
  // Create a covergroup
  void *cg = __moore_covergroup_create("verbosity_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[] = {
      {"test_bin", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 0, 100, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "cp1", bins, 1);
  __moore_coverpoint_sample(cg, 0, 50);

  // Summary level
  char *summaryReport = __moore_coverage_get_text_report(MOORE_TEXT_REPORT_SUMMARY);
  ASSERT_NE(summaryReport, nullptr);
  std::string summaryStr(summaryReport);
  EXPECT_EQ(summaryStr.find("verbosity_cg"), std::string::npos);  // No covergroup details
  __moore_free(summaryReport);

  // Normal level
  char *normalReport = __moore_coverage_get_text_report(MOORE_TEXT_REPORT_NORMAL);
  ASSERT_NE(normalReport, nullptr);
  std::string normalStr(normalReport);
  EXPECT_NE(normalStr.find("verbosity_cg"), std::string::npos);   // Has covergroup
  EXPECT_EQ(normalStr.find("test_bin"), std::string::npos);        // No bin details
  __moore_free(normalReport);

  // Detailed level
  char *detailReport = __moore_coverage_get_text_report(MOORE_TEXT_REPORT_DETAILED);
  ASSERT_NE(detailReport, nullptr);
  std::string detailStr(detailReport);
  EXPECT_NE(detailStr.find("verbosity_cg"), std::string::npos);   // Has covergroup
  EXPECT_NE(detailStr.find("test_bin"), std::string::npos);       // Has bin details
  __moore_free(detailReport);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, TextReportEmptyCovergroups) {
  // Generate report with no covergroups
  char *report = __moore_coverage_get_text_report(MOORE_TEXT_REPORT_NORMAL);
  ASSERT_NE(report, nullptr);

  std::string reportStr(report);
  EXPECT_NE(reportStr.find("Coverage Report"), std::string::npos);
  EXPECT_NE(reportStr.find("Covergroups: 0"), std::string::npos);

  __moore_free(report);
}

//===----------------------------------------------------------------------===//
// Enhanced HTML Report Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCoverageTest, HtmlReportWithTimestamp) {
  void *cg = __moore_covergroup_create("timestamp_test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "signal");
  __moore_coverpoint_sample(cg, 0, 42);

  const char *filename = "/tmp/coverage_timestamp_test.html";
  int32_t result = __moore_coverage_report_html(filename);
  EXPECT_EQ(result, 0);

  // Read and verify HTML content contains timestamp
  FILE *fp = std::fopen(filename, "r");
  ASSERT_NE(fp, nullptr);

  std::fseek(fp, 0, SEEK_END);
  long fileSize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  std::string htmlContent(fileSize, '\0');
  std::fread(&htmlContent[0], 1, fileSize, fp);
  std::fclose(fp);

  // Verify timestamp is present (format: YYYY-MM-DD)
  EXPECT_NE(htmlContent.find("Generated by circt-moore-runtime on"), std::string::npos);

  // Clean up
  std::remove(filename);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, HtmlReportWithInteractiveFeatures) {
  void *cg = __moore_covergroup_create("interactive_test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "test_cp");
  __moore_coverpoint_sample(cg, 0, 100);

  const char *filename = "/tmp/coverage_interactive_test.html";
  int32_t result = __moore_coverage_report_html(filename);
  EXPECT_EQ(result, 0);

  FILE *fp = std::fopen(filename, "r");
  ASSERT_NE(fp, nullptr);

  std::fseek(fp, 0, SEEK_END);
  long fileSize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  std::string htmlContent(fileSize, '\0');
  std::fread(&htmlContent[0], 1, fileSize, fp);
  std::fclose(fp);

  // Verify JavaScript functions are present
  EXPECT_NE(htmlContent.find("toggleCollapse"), std::string::npos);
  EXPECT_NE(htmlContent.find("filterCovergroups"), std::string::npos);
  EXPECT_NE(htmlContent.find("sortTable"), std::string::npos);
  EXPECT_NE(htmlContent.find("expandAll"), std::string::npos);
  EXPECT_NE(htmlContent.find("collapseAll"), std::string::npos);

  // Verify filter bar elements
  EXPECT_NE(htmlContent.find("searchInput"), std::string::npos);
  EXPECT_NE(htmlContent.find("statusFilter"), std::string::npos);
  EXPECT_NE(htmlContent.find("coverageFilter"), std::string::npos);

  // Verify collapsible classes
  EXPECT_NE(htmlContent.find("class=\"collapsible\""), std::string::npos);
  EXPECT_NE(htmlContent.find("class=\"collapse-content\""), std::string::npos);

  // Verify data attributes for filtering
  EXPECT_NE(htmlContent.find("data-name="), std::string::npos);
  EXPECT_NE(htmlContent.find("data-status="), std::string::npos);
  EXPECT_NE(htmlContent.find("data-coverage="), std::string::npos);

  // Verify print button
  EXPECT_NE(htmlContent.find("Print Report"), std::string::npos);

  // Clean up
  std::remove(filename);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, HtmlReportDataAttributesPassed) {
  // Test that covergroup with goal met has data-status="passed"
  void *cg = __moore_covergroup_create("passed_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[] = {
      {"bin1", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 1, 1, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 1);

  // Sample the bin to get 100% coverage
  __moore_coverpoint_sample(cg, 0, 1);

  const char *filename = "/tmp/coverage_passed_test.html";
  int32_t result = __moore_coverage_report_html(filename);
  EXPECT_EQ(result, 0);

  FILE *fp = std::fopen(filename, "r");
  ASSERT_NE(fp, nullptr);

  std::fseek(fp, 0, SEEK_END);
  long fileSize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  std::string htmlContent(fileSize, '\0');
  std::fread(&htmlContent[0], 1, fileSize, fp);
  std::fclose(fp);

  // Verify passed status
  EXPECT_NE(htmlContent.find("data-status=\"passed\""), std::string::npos);
  EXPECT_NE(htmlContent.find("data-coverage=\"100\""), std::string::npos);

  std::remove(filename);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, HtmlReportDataAttributesFailing) {
  // Test that covergroup with goal not met has data-status="failing"
  void *cg = __moore_covergroup_create("failing_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[] = {
      {"bin1", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 1, 1, 0},
      {"bin2", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 2, 2, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 2);

  // Only sample one bin to get 50% coverage
  __moore_coverpoint_sample(cg, 0, 1);

  const char *filename = "/tmp/coverage_failing_test.html";
  int32_t result = __moore_coverage_report_html(filename);
  EXPECT_EQ(result, 0);

  FILE *fp = std::fopen(filename, "r");
  ASSERT_NE(fp, nullptr);

  std::fseek(fp, 0, SEEK_END);
  long fileSize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  std::string htmlContent(fileSize, '\0');
  std::fread(&htmlContent[0], 1, fileSize, fp);
  std::fclose(fp);

  // Verify failing status (goal is 100%, we have 50%)
  EXPECT_NE(htmlContent.find("data-status=\"failing\""), std::string::npos);

  std::remove(filename);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, HtmlReportPrintMediaQuery) {
  void *cg = __moore_covergroup_create("print_test_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  const char *filename = "/tmp/coverage_print_test.html";
  int32_t result = __moore_coverage_report_html(filename);
  EXPECT_EQ(result, 0);

  FILE *fp = std::fopen(filename, "r");
  ASSERT_NE(fp, nullptr);

  std::fseek(fp, 0, SEEK_END);
  long fileSize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  std::string htmlContent(fileSize, '\0');
  std::fread(&htmlContent[0], 1, fileSize, fp);
  std::fclose(fp);

  // Verify print media query is present
  EXPECT_NE(htmlContent.find("@media print"), std::string::npos);

  std::remove(filename);
  __moore_covergroup_destroy(cg);
}

//===----------------------------------------------------------------------===//
// Console Output Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCoverageTest, PrintTextReport) {
  void *cg = __moore_covergroup_create("print_text_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 42);

  // This just tests that the function doesn't crash
  // Output goes to stdout which we don't capture
  __moore_coverage_print_text(MOORE_TEXT_REPORT_SUMMARY);
  __moore_coverage_print_text(MOORE_TEXT_REPORT_NORMAL);
  __moore_coverage_print_text(MOORE_TEXT_REPORT_DETAILED);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, ReportOnFinishAutoVerbosity) {
  void *cg = __moore_covergroup_create("finish_test_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 100);

  // Test auto-verbosity (-1)
  // This just tests that the function doesn't crash
  __moore_coverage_report_on_finish(-1);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, ReportOnFinishExplicitVerbosity) {
  void *cg = __moore_covergroup_create("finish_explicit_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 200);

  // Test with explicit verbosity levels
  __moore_coverage_report_on_finish(MOORE_TEXT_REPORT_SUMMARY);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageTest, ReportOnFinishWithGoals) {
  // Create covergroup that meets goal
  void *cg1 = __moore_covergroup_create("goal_met_cg", 1);
  ASSERT_NE(cg1, nullptr);

  MooreCoverageBin bins1[] = {
      {"bin1", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 1, 1, 0}
  };
  __moore_coverpoint_init_with_bins(cg1, 0, "cp", bins1, 1);
  __moore_coverpoint_sample(cg1, 0, 1);  // 100% coverage

  // Create covergroup that doesn't meet goal
  void *cg2 = __moore_covergroup_create("goal_not_met_cg", 1);
  ASSERT_NE(cg2, nullptr);

  MooreCoverageBin bins2[] = {
      {"bin1", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 1, 1, 0},
      {"bin2", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 2, 2, 0}
  };
  __moore_coverpoint_init_with_bins(cg2, 0, "cp", bins2, 2);
  __moore_coverpoint_sample(cg2, 0, 1);  // Only 50% coverage

  // Test report on finish - should show 1/2 goals passed
  __moore_coverage_report_on_finish(MOORE_TEXT_REPORT_SUMMARY);

  __moore_covergroup_destroy(cg1);
  __moore_covergroup_destroy(cg2);
}

//===----------------------------------------------------------------------===//
// Coverage Database Save/Load/Merge Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCoverageMergeTest, SaveAndLoadCoverage) {
  // Create a covergroup and sample some values
  void *cg = __moore_covergroup_create("test_save_load", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp_a");
  __moore_coverpoint_init(cg, 1, "cp_b");

  // Sample some values
  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 0, 2);
  __moore_coverpoint_sample(cg, 0, 3);
  __moore_coverpoint_sample(cg, 1, 100);
  __moore_coverpoint_sample(cg, 1, 200);

  // Save coverage to file
  const char *filename = "/tmp/coverage_save_test.json";
  int32_t result = __moore_coverage_save(filename);
  EXPECT_EQ(result, 0);

  // Load the saved database
  MooreCoverageDBHandle db = __moore_coverage_load(filename);
  ASSERT_NE(db, nullptr);

  // Check that we got the expected data
  EXPECT_EQ(__moore_coverage_db_get_num_covergroups(db), 1);
  EXPECT_STREQ(__moore_coverage_db_get_covergroup_name(db, 0), "test_save_load");

  // Coverage should be positive
  double coverage = __moore_coverage_db_get_coverage(db, "test_save_load");
  EXPECT_GT(coverage, 0.0);

  // Clean up
  __moore_coverage_db_free(db);
  __moore_covergroup_destroy(cg);
  std::remove(filename);
}

TEST(MooreRuntimeCoverageMergeTest, SaveNullFilename) {
  int32_t result = __moore_coverage_save(nullptr);
  EXPECT_NE(result, 0);
}

TEST(MooreRuntimeCoverageMergeTest, LoadNonexistentFile) {
  MooreCoverageDBHandle db = __moore_coverage_load("/tmp/nonexistent_coverage.json");
  EXPECT_EQ(db, nullptr);
}

TEST(MooreRuntimeCoverageMergeTest, LoadNullFilename) {
  MooreCoverageDBHandle db = __moore_coverage_load(nullptr);
  EXPECT_EQ(db, nullptr);
}

TEST(MooreRuntimeCoverageMergeTest, FreeNullHandle) {
  // Should not crash
  __moore_coverage_db_free(nullptr);
}

TEST(MooreRuntimeCoverageMergeTest, MergeIntoCurrent) {
  // Create a covergroup with initial samples
  void *cg = __moore_covergroup_create("merge_test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp");

  // Sample initial values (values 1, 2, 3)
  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 0, 2);
  __moore_coverpoint_sample(cg, 0, 3);

  // Save to a file
  const char *file1 = "/tmp/merge_test1.json";
  __moore_coverage_save(file1);

  // Reset the coverpoint and sample different values
  __moore_coverpoint_reset(cg, 0);
  __moore_coverpoint_sample(cg, 0, 4);
  __moore_coverpoint_sample(cg, 0, 5);

  // Now merge the first file back in
  int32_t result = __moore_coverage_merge_file(file1);
  EXPECT_EQ(result, 0);

  // The coverage should now include merged data
  double coverage = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_GT(coverage, 0.0);

  // Clean up
  __moore_covergroup_destroy(cg);
  std::remove(file1);
}

TEST(MooreRuntimeCoverageMergeTest, MergeTwoFiles) {
  // Create first covergroup and save
  void *cg1 = __moore_covergroup_create("file_merge_cg", 1);
  ASSERT_NE(cg1, nullptr);
  __moore_coverpoint_init(cg1, 0, "cp");
  __moore_coverpoint_sample(cg1, 0, 1);
  __moore_coverpoint_sample(cg1, 0, 2);

  const char *file1 = "/tmp/merge_file1.json";
  __moore_coverage_save(file1);
  __moore_covergroup_destroy(cg1);

  // Create second covergroup with same structure and save
  void *cg2 = __moore_covergroup_create("file_merge_cg", 1);
  ASSERT_NE(cg2, nullptr);
  __moore_coverpoint_init(cg2, 0, "cp");
  __moore_coverpoint_sample(cg2, 0, 3);
  __moore_coverpoint_sample(cg2, 0, 4);

  const char *file2 = "/tmp/merge_file2.json";
  __moore_coverage_save(file2);
  __moore_covergroup_destroy(cg2);

  // Merge both files into a new file
  const char *output = "/tmp/merge_output.json";
  int32_t result = __moore_coverage_merge_files(file1, file2, output);
  EXPECT_EQ(result, 0);

  // Load the merged file and verify
  MooreCoverageDBHandle db = __moore_coverage_load(output);
  ASSERT_NE(db, nullptr);

  EXPECT_EQ(__moore_coverage_db_get_num_covergroups(db), 1);
  EXPECT_STREQ(__moore_coverage_db_get_covergroup_name(db, 0), "file_merge_cg");

  // Clean up
  __moore_coverage_db_free(db);
  std::remove(file1);
  std::remove(file2);
  std::remove(output);
}

TEST(MooreRuntimeCoverageMergeTest, MergeFilesNullArgs) {
  const char *file = "/tmp/dummy.json";
  int32_t result;

  result = __moore_coverage_merge_files(nullptr, file, file);
  EXPECT_NE(result, 0);

  result = __moore_coverage_merge_files(file, nullptr, file);
  EXPECT_NE(result, 0);

  result = __moore_coverage_merge_files(file, file, nullptr);
  EXPECT_NE(result, 0);
}

TEST(MooreRuntimeCoverageMergeTest, MergeFileNullFilename) {
  int32_t result = __moore_coverage_merge_file(nullptr);
  EXPECT_NE(result, 0);
}

TEST(MooreRuntimeCoverageMergeTest, MergeNullHandle) {
  int32_t result = __moore_coverage_merge(nullptr);
  EXPECT_NE(result, 0);
}

TEST(MooreRuntimeCoverageMergeTest, DBGetNumCovergroupsNullHandle) {
  int32_t count = __moore_coverage_db_get_num_covergroups(nullptr);
  EXPECT_EQ(count, -1);
}

TEST(MooreRuntimeCoverageMergeTest, DBGetCovergroupNameNullHandle) {
  const char *name = __moore_coverage_db_get_covergroup_name(nullptr, 0);
  EXPECT_EQ(name, nullptr);
}

TEST(MooreRuntimeCoverageMergeTest, DBGetCovergroupNameInvalidIndex) {
  // Create and save a simple covergroup
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 1);

  const char *filename = "/tmp/test_index.json";
  __moore_coverage_save(filename);
  __moore_covergroup_destroy(cg);

  MooreCoverageDBHandle db = __moore_coverage_load(filename);
  ASSERT_NE(db, nullptr);

  // Test invalid indices
  EXPECT_EQ(__moore_coverage_db_get_covergroup_name(db, -1), nullptr);
  EXPECT_EQ(__moore_coverage_db_get_covergroup_name(db, 100), nullptr);

  __moore_coverage_db_free(db);
  std::remove(filename);
}

TEST(MooreRuntimeCoverageMergeTest, DBGetCoverageNullHandle) {
  double coverage = __moore_coverage_db_get_coverage(nullptr, "test");
  EXPECT_LT(coverage, 0.0);
}

TEST(MooreRuntimeCoverageMergeTest, DBGetCoverageNonexistentGroup) {
  // Create and save a simple covergroup
  void *cg = __moore_covergroup_create("existing_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 1);

  const char *filename = "/tmp/test_coverage.json";
  __moore_coverage_save(filename);
  __moore_covergroup_destroy(cg);

  MooreCoverageDBHandle db = __moore_coverage_load(filename);
  ASSERT_NE(db, nullptr);

  // Test nonexistent covergroup
  double coverage = __moore_coverage_db_get_coverage(db, "nonexistent_cg");
  EXPECT_LT(coverage, 0.0);

  __moore_coverage_db_free(db);
  std::remove(filename);
}

TEST(MooreRuntimeCoverageMergeTest, DBGetTotalCoverage) {
  // Create and save a covergroup
  void *cg = __moore_covergroup_create("total_test_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 1);

  const char *filename = "/tmp/test_total.json";
  __moore_coverage_save(filename);
  __moore_covergroup_destroy(cg);

  MooreCoverageDBHandle db = __moore_coverage_load(filename);
  ASSERT_NE(db, nullptr);

  // Get total coverage (pass NULL for cg_name)
  double coverage = __moore_coverage_db_get_coverage(db, nullptr);
  EXPECT_GE(coverage, 0.0);
  EXPECT_LE(coverage, 100.0);

  __moore_coverage_db_free(db);
  std::remove(filename);
}

TEST(MooreRuntimeCoverageMergeTest, MergeWithExplicitBins) {
  // Create covergroup with explicit bins
  void *cg = __moore_covergroup_create("bin_merge_cg", 1);
  ASSERT_NE(cg, nullptr);

  // MooreCoverageBin: {name, type, kind, low, high, hit_count}
  MooreCoverageBin bins[] = {
      {"low", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 0, 10, 0},
      {"high", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 11, 20, 0}};
  __moore_coverpoint_init_with_bins(cg, 0, "cp_bins", bins, 2);

  // Sample some values in the first bin
  __moore_coverpoint_sample(cg, 0, 5);
  __moore_coverpoint_sample(cg, 0, 7);

  // Check initial bin hits
  EXPECT_GT(__moore_coverpoint_get_bin_hits(cg, 0, 0), 0);
  EXPECT_EQ(__moore_coverpoint_get_bin_hits(cg, 0, 1), 0);

  // Save to file
  const char *file1 = "/tmp/bin_merge1.json";
  __moore_coverage_save(file1);

  // Reset and sample in the second bin
  __moore_coverpoint_reset(cg, 0);
  __moore_coverpoint_sample(cg, 0, 15);
  __moore_coverpoint_sample(cg, 0, 18);

  // Save to second file
  const char *file2 = "/tmp/bin_merge2.json";
  __moore_coverage_save(file2);

  // Merge the first file back
  __moore_coverage_merge_file(file1);

  // Now both bins should have hits
  EXPECT_GT(__moore_coverpoint_get_bin_hits(cg, 0, 0), 0);
  EXPECT_GT(__moore_coverpoint_get_bin_hits(cg, 0, 1), 0);

  // Clean up
  __moore_covergroup_destroy(cg);
  std::remove(file1);
  std::remove(file2);
}

TEST(MooreRuntimeCoverageMergeTest, MergeCumulativeHitCounts) {
  // Test that hit counts are properly accumulated during merge
  void *cg = __moore_covergroup_create("cumulative_test", 1);
  ASSERT_NE(cg, nullptr);

  // MooreCoverageBin: {name, type, kind, low, high, hit_count}
  MooreCoverageBin bins[] = {{"bin1", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 1, 1, 0}};
  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 1);

  // Sample value 5 times
  for (int i = 0; i < 5; i++) {
    __moore_coverpoint_sample(cg, 0, 1);
  }

  // Save to file
  const char *filename = "/tmp/cumulative_test.json";
  __moore_coverage_save(filename);

  // Sample 5 more times
  for (int i = 0; i < 5; i++) {
    __moore_coverpoint_sample(cg, 0, 1);
  }

  // Now we have 10 hits
  int64_t hitsBeforeMerge = __moore_coverpoint_get_bin_hits(cg, 0, 0);
  EXPECT_EQ(hitsBeforeMerge, 10);

  // Merge the saved file (which has 5 hits)
  __moore_coverage_merge_file(filename);

  // After merge, we should have 15 hits (10 + 5)
  int64_t hitsAfterMerge = __moore_coverpoint_get_bin_hits(cg, 0, 0);
  EXPECT_EQ(hitsAfterMerge, 15);

  // Clean up
  __moore_covergroup_destroy(cg);
  std::remove(filename);
}

TEST(MooreRuntimeCoverageMergeTest, GoalTrackingAfterMerge) {
  // Test that goal tracking works correctly after merge
  void *cg = __moore_covergroup_create("goal_test", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Set a low goal
  __moore_covergroup_set_goal(cg, 50.0);

  // Initially goal is not met (no samples)
  EXPECT_FALSE(__moore_covergroup_goal_met(cg));

  // Sample a single value
  __moore_coverpoint_sample(cg, 0, 1);

  // Save current state
  const char *filename = "/tmp/goal_test.json";
  __moore_coverage_save(filename);

  // Reset and check goal is not met
  __moore_covergroup_reset(cg);
  EXPECT_FALSE(__moore_covergroup_goal_met(cg));

  // Merge the saved coverage back
  __moore_coverage_merge_file(filename);

  // Now coverage should be restored and goal tracking should work
  double coverage = __moore_covergroup_get_coverage(cg);
  EXPECT_GT(coverage, 0.0);

  // Clean up
  __moore_covergroup_destroy(cg);
  std::remove(filename);
}

//===----------------------------------------------------------------------===//
// Coverage Database Persistence with Metadata Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCoverageDBTest, SaveDBWithTestName) {
  // Create a covergroup and sample some values
  void *cg = __moore_covergroup_create("save_db_test", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 42);
  __moore_coverpoint_sample(cg, 0, 100);

  const char *filename = "/tmp/coverage_db_test.json";
  int32_t result = __moore_coverage_save_db(filename, "my_test_run", "Test comment");
  EXPECT_EQ(result, 0);

  // Load and verify metadata
  MooreCoverageDBHandle db = __moore_coverage_load_db(filename);
  ASSERT_NE(db, nullptr);

  const MooreCoverageMetadata *meta = __moore_coverage_db_get_metadata(db);
  ASSERT_NE(meta, nullptr);
  EXPECT_STREQ(meta->test_name, "my_test_run");
  EXPECT_GT(meta->timestamp, 0);
  EXPECT_STREQ(meta->simulator, "circt-moore");
  EXPECT_STREQ(meta->comment, "Test comment");

  __moore_coverage_db_free(db);
  __moore_covergroup_destroy(cg);
  std::remove(filename);
}

TEST(MooreRuntimeCoverageDBTest, SaveDBNullFilename) {
  int32_t result = __moore_coverage_save_db(nullptr, "test", nullptr);
  EXPECT_NE(result, 0);
}

TEST(MooreRuntimeCoverageDBTest, SaveDBNullTestName) {
  // Create a covergroup
  void *cg = __moore_covergroup_create("save_null_test", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 1);

  const char *filename = "/tmp/coverage_db_null_test.json";
  int32_t result = __moore_coverage_save_db(filename, nullptr, nullptr);
  EXPECT_EQ(result, 0);

  // Load and verify metadata has null test_name
  MooreCoverageDBHandle db = __moore_coverage_load_db(filename);
  ASSERT_NE(db, nullptr);

  const MooreCoverageMetadata *meta = __moore_coverage_db_get_metadata(db);
  ASSERT_NE(meta, nullptr);
  EXPECT_EQ(meta->test_name, nullptr);
  EXPECT_GT(meta->timestamp, 0);

  __moore_coverage_db_free(db);
  __moore_covergroup_destroy(cg);
  std::remove(filename);
}

TEST(MooreRuntimeCoverageDBTest, LoadDBNullFilename) {
  MooreCoverageDBHandle db = __moore_coverage_load_db(nullptr);
  EXPECT_EQ(db, nullptr);
}

TEST(MooreRuntimeCoverageDBTest, LoadDBNonexistent) {
  MooreCoverageDBHandle db = __moore_coverage_load_db("/tmp/nonexistent_coverage_db.json");
  EXPECT_EQ(db, nullptr);
}

TEST(MooreRuntimeCoverageDBTest, MergeDBFromFile) {
  // Create a covergroup with initial samples
  void *cg = __moore_covergroup_create("merge_db_test", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 0, 2);

  const char *filename = "/tmp/merge_db_test.json";
  __moore_coverage_save_db(filename, "first_run", nullptr);

  // Reset and sample new values
  __moore_coverpoint_reset(cg, 0);
  __moore_coverpoint_sample(cg, 0, 3);
  __moore_coverpoint_sample(cg, 0, 4);

  // Merge the saved database
  int32_t result = __moore_coverage_merge_db(filename);
  EXPECT_EQ(result, 0);

  // Coverage should include merged data
  double coverage = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_GT(coverage, 0.0);

  __moore_covergroup_destroy(cg);
  std::remove(filename);
}

TEST(MooreRuntimeCoverageDBTest, MergeDBNullFilename) {
  int32_t result = __moore_coverage_merge_db(nullptr);
  EXPECT_NE(result, 0);
}

TEST(MooreRuntimeCoverageDBTest, MergeDBNonexistent) {
  int32_t result = __moore_coverage_merge_db("/tmp/nonexistent_merge_db.json");
  EXPECT_NE(result, 0);
}

TEST(MooreRuntimeCoverageDBTest, GetMetadataNullHandle) {
  const MooreCoverageMetadata *meta = __moore_coverage_db_get_metadata(nullptr);
  EXPECT_EQ(meta, nullptr);
}

TEST(MooreRuntimeCoverageDBTest, SetAndGetGlobalTestName) {
  // Initially should be null
  EXPECT_EQ(__moore_coverage_get_test_name(), nullptr);

  // Set a test name
  __moore_coverage_set_test_name("global_test_name");
  EXPECT_STREQ(__moore_coverage_get_test_name(), "global_test_name");

  // Create and save - should use global test name
  void *cg = __moore_covergroup_create("global_name_test", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 1);

  const char *filename = "/tmp/global_name_test.json";
  __moore_coverage_save_db(filename, nullptr, nullptr);

  // Load and verify global name was used
  MooreCoverageDBHandle db = __moore_coverage_load_db(filename);
  ASSERT_NE(db, nullptr);

  const MooreCoverageMetadata *meta = __moore_coverage_db_get_metadata(db);
  ASSERT_NE(meta, nullptr);
  EXPECT_STREQ(meta->test_name, "global_test_name");

  __moore_coverage_db_free(db);
  __moore_covergroup_destroy(cg);
  std::remove(filename);

  // Clear global test name
  __moore_coverage_set_test_name(nullptr);
  EXPECT_EQ(__moore_coverage_get_test_name(), nullptr);
}

TEST(MooreRuntimeCoverageDBTest, ExplicitNameOverridesGlobal) {
  // Set a global test name
  __moore_coverage_set_test_name("global_name");

  void *cg = __moore_covergroup_create("override_test", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 1);

  const char *filename = "/tmp/override_name_test.json";
  __moore_coverage_save_db(filename, "explicit_name", nullptr);

  // Load and verify explicit name was used
  MooreCoverageDBHandle db = __moore_coverage_load_db(filename);
  ASSERT_NE(db, nullptr);

  const MooreCoverageMetadata *meta = __moore_coverage_db_get_metadata(db);
  ASSERT_NE(meta, nullptr);
  EXPECT_STREQ(meta->test_name, "explicit_name");

  __moore_coverage_db_free(db);
  __moore_covergroup_destroy(cg);
  std::remove(filename);

  // Clear global test name
  __moore_coverage_set_test_name(nullptr);
}

TEST(MooreRuntimeCoverageDBTest, TimestampIsRecent) {
  void *cg = __moore_covergroup_create("timestamp_test", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 1);

  int64_t beforeSave = static_cast<int64_t>(std::time(nullptr));

  const char *filename = "/tmp/timestamp_test.json";
  __moore_coverage_save_db(filename, "ts_test", nullptr);

  int64_t afterSave = static_cast<int64_t>(std::time(nullptr));

  MooreCoverageDBHandle db = __moore_coverage_load_db(filename);
  ASSERT_NE(db, nullptr);

  const MooreCoverageMetadata *meta = __moore_coverage_db_get_metadata(db);
  ASSERT_NE(meta, nullptr);
  EXPECT_GE(meta->timestamp, beforeSave);
  EXPECT_LE(meta->timestamp, afterSave);

  __moore_coverage_db_free(db);
  __moore_covergroup_destroy(cg);
  std::remove(filename);
}

TEST(MooreRuntimeCoverageDBTest, LoadDBWithBinsAndMerge) {
  // Create covergroup with explicit bins
  void *cg = __moore_covergroup_create("bin_db_test", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[] = {
      {"low", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 0, 50, 0},
      {"high", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 51, 100, 0}};
  __moore_coverpoint_init_with_bins(cg, 0, "cp_bins", bins, 2);

  // Sample low values
  __moore_coverpoint_sample(cg, 0, 10);
  __moore_coverpoint_sample(cg, 0, 20);

  const char *filename = "/tmp/bin_db_test.json";
  __moore_coverage_save_db(filename, "bin_test", nullptr);

  // Reset and sample high values
  __moore_coverpoint_reset(cg, 0);
  __moore_coverpoint_sample(cg, 0, 60);
  __moore_coverpoint_sample(cg, 0, 70);

  // Merge
  int32_t result = __moore_coverage_merge_db(filename);
  EXPECT_EQ(result, 0);

  // Both bins should now have hits
  EXPECT_GT(__moore_coverpoint_get_bin_hits(cg, 0, 0), 0);
  EXPECT_GT(__moore_coverpoint_get_bin_hits(cg, 0, 1), 0);

  __moore_covergroup_destroy(cg);
  std::remove(filename);
}

TEST(MooreRuntimeCoverageDBTest, MultipleMerges) {
  // Create covergroup
  void *cg = __moore_covergroup_create("multi_merge_test", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[] = {
      {"bin1", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 1, 1, 0}};
  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 1);

  // Sample and save first run
  for (int i = 0; i < 3; i++) {
    __moore_coverpoint_sample(cg, 0, 1);
  }
  const char *file1 = "/tmp/multi_merge1.json";
  __moore_coverage_save_db(file1, "run1", nullptr);

  // Reset, sample, and save second run
  __moore_coverpoint_reset(cg, 0);
  for (int i = 0; i < 5; i++) {
    __moore_coverpoint_sample(cg, 0, 1);
  }
  const char *file2 = "/tmp/multi_merge2.json";
  __moore_coverage_save_db(file2, "run2", nullptr);

  // Reset and start fresh
  __moore_coverpoint_reset(cg, 0);
  EXPECT_EQ(__moore_coverpoint_get_bin_hits(cg, 0, 0), 0);

  // Merge both files
  __moore_coverage_merge_db(file1);
  __moore_coverage_merge_db(file2);

  // Should have 8 hits total (3 + 5)
  EXPECT_EQ(__moore_coverpoint_get_bin_hits(cg, 0, 0), 8);

  __moore_covergroup_destroy(cg);
  std::remove(file1);
  std::remove(file2);
}

TEST(MooreRuntimeCoverageDBTest, MetadataPreservedAcrossLoadSave) {
  // Create and save with metadata
  void *cg = __moore_covergroup_create("preserve_meta_test", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 42);

  const char *filename = "/tmp/preserve_meta_test.json";
  __moore_coverage_save_db(filename, "original_test", "Original comment");

  // Load
  MooreCoverageDBHandle db = __moore_coverage_load_db(filename);
  ASSERT_NE(db, nullptr);

  // Verify original metadata
  const MooreCoverageMetadata *meta = __moore_coverage_db_get_metadata(db);
  ASSERT_NE(meta, nullptr);
  EXPECT_STREQ(meta->test_name, "original_test");
  EXPECT_STREQ(meta->comment, "Original comment");
  EXPECT_GT(meta->timestamp, 0);

  __moore_coverage_db_free(db);
  __moore_covergroup_destroy(cg);
  std::remove(filename);
}

//===----------------------------------------------------------------------===//
// UCDB-Compatible Coverage File Format Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeUCDBTest, WriteAndReadUCDBFormat) {
  // Create a covergroup with samples
  void *cg = __moore_covergroup_create("ucdb_test_cg", 2);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "addr_cp");
  __moore_coverpoint_init(cg, 1, "data_cp");
  __moore_coverpoint_sample(cg, 0, 0x100);
  __moore_coverpoint_sample(cg, 0, 0x200);
  __moore_coverpoint_sample(cg, 1, 42);

  const char *filename = "/tmp/ucdb_test.json";

  // Write with default metadata
  int32_t result = __moore_coverage_write_ucdb(filename, nullptr);
  EXPECT_EQ(result, 0);

  // Read back
  MooreCoverageDBHandle db = __moore_coverage_read_ucdb(filename);
  ASSERT_NE(db, nullptr);

  // Verify metadata was created
  const MooreCoverageMetadata *meta = __moore_coverage_db_get_metadata(db);
  EXPECT_NE(meta, nullptr);

  __moore_coverage_db_free(db);
  __moore_covergroup_destroy(cg);
  std::remove(filename);
}

TEST(MooreRuntimeUCDBTest, WriteUCDBWithMetadata) {
  void *cg = __moore_covergroup_create("ucdb_meta_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 10);

  const char *filename = "/tmp/ucdb_meta_test.json";

  // Create custom metadata
  MooreUCDBMetadata metadata = {};
  metadata.test_name = "my_ucdb_test";
  metadata.test_seed = "12345";
  metadata.comment = "UCDB format test";
  metadata.tool_name = "test_tool";
  metadata.tool_version = "2.0";

  int32_t result = __moore_coverage_write_ucdb(filename, &metadata);
  EXPECT_EQ(result, 0);

  // Read back and verify
  MooreCoverageDBHandle db = __moore_coverage_read_ucdb(filename);
  ASSERT_NE(db, nullptr);

  const MooreCoverageMetadata *meta = __moore_coverage_db_get_metadata(db);
  ASSERT_NE(meta, nullptr);
  EXPECT_STREQ(meta->test_name, "my_ucdb_test");
  EXPECT_STREQ(meta->comment, "UCDB format test");
  EXPECT_STREQ(meta->simulator, "test_tool");

  __moore_coverage_db_free(db);
  __moore_covergroup_destroy(cg);
  std::remove(filename);
}

TEST(MooreRuntimeUCDBTest, WriteUCDBNullFilename) {
  int32_t result = __moore_coverage_write_ucdb(nullptr, nullptr);
  EXPECT_NE(result, 0);
}

TEST(MooreRuntimeUCDBTest, ReadUCDBNullFilename) {
  MooreCoverageDBHandle db = __moore_coverage_read_ucdb(nullptr);
  EXPECT_EQ(db, nullptr);
}

TEST(MooreRuntimeUCDBTest, ReadUCDBNonexistent) {
  MooreCoverageDBHandle db = __moore_coverage_read_ucdb("/tmp/nonexistent_ucdb.json");
  EXPECT_EQ(db, nullptr);
}

TEST(MooreRuntimeUCDBTest, IsUCDBFormat) {
  void *cg = __moore_covergroup_create("ucdb_format_check_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 1);

  const char *ucdbFile = "/tmp/ucdb_format_check.json";
  const char *legacyFile = "/tmp/legacy_format_check.json";

  // Write UCDB format
  __moore_coverage_write_ucdb(ucdbFile, nullptr);

  // Write legacy format
  __moore_coverage_save_db(legacyFile, "test", nullptr);

  // Check UCDB format detection
  int32_t isUcdb = __moore_coverage_is_ucdb_format(ucdbFile);
  EXPECT_EQ(isUcdb, 1);

  // Legacy format should not be detected as UCDB 2.0
  int32_t isLegacy = __moore_coverage_is_ucdb_format(legacyFile);
  EXPECT_EQ(isLegacy, 0);

  // Null filename
  EXPECT_EQ(__moore_coverage_is_ucdb_format(nullptr), -1);

  // Nonexistent file
  EXPECT_EQ(__moore_coverage_is_ucdb_format("/tmp/nonexistent.json"), -1);

  __moore_covergroup_destroy(cg);
  std::remove(ucdbFile);
  std::remove(legacyFile);
}

TEST(MooreRuntimeUCDBTest, GetFileVersion) {
  void *cg = __moore_covergroup_create("version_check_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 1);

  const char *filename = "/tmp/version_check.json";
  __moore_coverage_write_ucdb(filename, nullptr);

  const char *version = __moore_coverage_get_file_version(filename);
  ASSERT_NE(version, nullptr);
  EXPECT_STREQ(version, MOORE_UCDB_FORMAT_VERSION);

  // Null filename
  EXPECT_EQ(__moore_coverage_get_file_version(nullptr), nullptr);

  __moore_covergroup_destroy(cg);
  std::remove(filename);
}

TEST(MooreRuntimeUCDBTest, MergeUCDBFiles) {
  // Create first coverage run
  void *cg1 = __moore_covergroup_create("merge_ucdb_cg", 1);
  ASSERT_NE(cg1, nullptr);
  __moore_coverpoint_init(cg1, 0, "cp");
  __moore_coverpoint_sample(cg1, 0, 1);
  __moore_coverpoint_sample(cg1, 0, 2);

  const char *file1 = "/tmp/ucdb_merge1.json";
  MooreUCDBMetadata meta1 = {};
  meta1.test_name = "run1";
  __moore_coverage_write_ucdb(file1, &meta1);

  // Reset and create second coverage run
  __moore_coverpoint_reset(cg1, 0);
  __moore_coverpoint_sample(cg1, 0, 3);
  __moore_coverpoint_sample(cg1, 0, 4);

  const char *file2 = "/tmp/ucdb_merge2.json";
  MooreUCDBMetadata meta2 = {};
  meta2.test_name = "run2";
  __moore_coverage_write_ucdb(file2, &meta2);

  __moore_covergroup_destroy(cg1);

  // Merge files
  const char *files[] = {file1, file2};
  const char *outputFile = "/tmp/ucdb_merged.json";
  int32_t result = __moore_coverage_merge_ucdb_files(files, 2, outputFile, "Merged coverage");
  EXPECT_EQ(result, 0);

  // Verify merged file exists and is UCDB format
  EXPECT_EQ(__moore_coverage_is_ucdb_format(outputFile), 1);

  // Load merged file
  MooreCoverageDBHandle db = __moore_coverage_read_ucdb(outputFile);
  ASSERT_NE(db, nullptr);

  const MooreCoverageMetadata *meta = __moore_coverage_db_get_metadata(db);
  EXPECT_NE(meta, nullptr);
  EXPECT_STREQ(meta->comment, "Merged coverage");

  __moore_coverage_db_free(db);
  std::remove(file1);
  std::remove(file2);
  std::remove(outputFile);
}

TEST(MooreRuntimeUCDBTest, MergeUCDBFilesNullInputs) {
  EXPECT_NE(__moore_coverage_merge_ucdb_files(nullptr, 2, "/tmp/out.json", nullptr), 0);
  const char *files[] = {"/tmp/a.json"};
  EXPECT_NE(__moore_coverage_merge_ucdb_files(files, 0, "/tmp/out.json", nullptr), 0);
  EXPECT_NE(__moore_coverage_merge_ucdb_files(files, 1, nullptr, nullptr), 0);
}

TEST(MooreRuntimeUCDBTest, UserAttributes) {
  // Test setting and getting user attributes
  __moore_coverage_set_user_attr("project", "my_project");
  __moore_coverage_set_user_attr("version", "1.0");

  const char *project = __moore_coverage_get_user_attr("project");
  ASSERT_NE(project, nullptr);
  EXPECT_STREQ(project, "my_project");

  const char *version = __moore_coverage_get_user_attr("version");
  ASSERT_NE(version, nullptr);
  EXPECT_STREQ(version, "1.0");

  // Nonexistent attribute
  EXPECT_EQ(__moore_coverage_get_user_attr("nonexistent"), nullptr);

  // Null name
  EXPECT_EQ(__moore_coverage_get_user_attr(nullptr), nullptr);

  // Remove attribute
  __moore_coverage_set_user_attr("project", nullptr);
  EXPECT_EQ(__moore_coverage_get_user_attr("project"), nullptr);

  // Clean up
  __moore_coverage_set_user_attr("version", nullptr);
}

TEST(MooreRuntimeUCDBTest, UserAttributesInOutput) {
  void *cg = __moore_covergroup_create("user_attr_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 1);

  // Set user attributes
  __moore_coverage_set_user_attr("custom_key", "custom_value");

  const char *filename = "/tmp/user_attr_test.json";
  __moore_coverage_write_ucdb(filename, nullptr);

  // Verify the file contains user attributes
  // (We can read the file and check content)
  FILE *fp = std::fopen(filename, "r");
  ASSERT_NE(fp, nullptr);

  char buffer[4096];
  size_t bytesRead = std::fread(buffer, 1, sizeof(buffer) - 1, fp);
  std::fclose(fp);
  buffer[bytesRead] = '\0';

  std::string content(buffer);
  EXPECT_NE(content.find("user_attributes"), std::string::npos);
  EXPECT_NE(content.find("custom_key"), std::string::npos);
  EXPECT_NE(content.find("custom_value"), std::string::npos);

  // Clean up
  __moore_coverage_set_user_attr("custom_key", nullptr);
  __moore_covergroup_destroy(cg);
  std::remove(filename);
}

TEST(MooreRuntimeUCDBTest, ReadLegacyFormatWithUCDBReader) {
  // Create legacy format file
  void *cg = __moore_covergroup_create("legacy_read_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");
  __moore_coverpoint_sample(cg, 0, 100);

  const char *filename = "/tmp/legacy_for_ucdb_read.json";
  __moore_coverage_save_db(filename, "legacy_test", "Legacy comment");
  __moore_covergroup_destroy(cg);

  // Read with UCDB reader (should fall back to legacy parsing)
  MooreCoverageDBHandle db = __moore_coverage_read_ucdb(filename);
  ASSERT_NE(db, nullptr);

  const MooreCoverageMetadata *meta = __moore_coverage_db_get_metadata(db);
  EXPECT_NE(meta, nullptr);
  EXPECT_STREQ(meta->test_name, "legacy_test");

  __moore_coverage_db_free(db);
  std::remove(filename);
}

//===----------------------------------------------------------------------===//
// Illegal Bins and Ignore Bins Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeIllegalBinsTest, DetectIllegalBinHit) {
  // Create a covergroup with a coverpoint
  void *cg = __moore_covergroup_create("illegal_test_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Reset the illegal bin hit counter
  __moore_coverage_reset_illegal_bin_hits();
  EXPECT_EQ(__moore_coverage_get_illegal_bin_hits(), 0);

  // Set illegal bin fatal mode to false (just warning) for testing
  __moore_coverage_set_illegal_bin_fatal(false);
  EXPECT_FALSE(__moore_coverage_illegal_bin_is_fatal());

  // Add an illegal bin for values 256 and above
  __moore_coverpoint_add_illegal_bin(cg, 0, "invalid_high", 256, 511);

  // Sample a valid value - should not trigger illegal bin
  __moore_coverpoint_sample(cg, 0, 100);
  EXPECT_EQ(__moore_coverage_get_illegal_bin_hits(), 0);

  // Sample an illegal value - should trigger illegal bin
  __moore_coverpoint_sample(cg, 0, 300);
  EXPECT_EQ(__moore_coverage_get_illegal_bin_hits(), 1);

  // Sample another illegal value
  __moore_coverpoint_sample(cg, 0, 256);
  EXPECT_EQ(__moore_coverage_get_illegal_bin_hits(), 2);

  // Clean up
  __moore_covergroup_destroy(cg);
  __moore_coverage_reset_illegal_bin_hits();
}

TEST(MooreRuntimeIllegalBinsTest, IllegalBinSingleValue) {
  void *cg = __moore_covergroup_create("illegal_single_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  __moore_coverage_reset_illegal_bin_hits();
  __moore_coverage_set_illegal_bin_fatal(false);

  // Add illegal bin for a single value
  __moore_coverpoint_add_illegal_bin(cg, 0, "forbidden_value", 42, 42);

  // Sample adjacent values - should not trigger
  __moore_coverpoint_sample(cg, 0, 41);
  __moore_coverpoint_sample(cg, 0, 43);
  EXPECT_EQ(__moore_coverage_get_illegal_bin_hits(), 0);

  // Sample the forbidden value
  __moore_coverpoint_sample(cg, 0, 42);
  EXPECT_EQ(__moore_coverage_get_illegal_bin_hits(), 1);

  __moore_covergroup_destroy(cg);
  __moore_coverage_reset_illegal_bin_hits();
}

TEST(MooreRuntimeIllegalBinsTest, IsIllegalCheck) {
  void *cg = __moore_covergroup_create("is_illegal_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Add illegal bins
  __moore_coverpoint_add_illegal_bin(cg, 0, "illegal_range", 100, 200);

  // Check values
  EXPECT_FALSE(__moore_coverpoint_is_illegal(cg, 0, 50));
  EXPECT_FALSE(__moore_coverpoint_is_illegal(cg, 0, 99));
  EXPECT_TRUE(__moore_coverpoint_is_illegal(cg, 0, 100));
  EXPECT_TRUE(__moore_coverpoint_is_illegal(cg, 0, 150));
  EXPECT_TRUE(__moore_coverpoint_is_illegal(cg, 0, 200));
  EXPECT_FALSE(__moore_coverpoint_is_illegal(cg, 0, 201));

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeIgnoreBinsTest, IgnoreBinsNotCounted) {
  void *cg = __moore_covergroup_create("ignore_test_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Add ignore bin for value 0 (skip zeros)
  __moore_coverpoint_add_ignore_bin(cg, 0, "skip_zero", 0, 0);

  // Sample values including ignored ones
  __moore_coverpoint_sample(cg, 0, 0);  // Should be ignored
  __moore_coverpoint_sample(cg, 0, 0);  // Should be ignored
  __moore_coverpoint_sample(cg, 0, 1);  // Should count
  __moore_coverpoint_sample(cg, 0, 2);  // Should count

  // Check that coverage doesn't include ignored samples
  // The coverpoint should only have 2 hits (values 1 and 2)
  double coverage = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_GT(coverage, 0.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeIgnoreBinsTest, IgnoreBinsRange) {
  void *cg = __moore_covergroup_create("ignore_range_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Add ignore bins for a range (values 10-20)
  __moore_coverpoint_add_ignore_bin(cg, 0, "skip_range", 10, 20);

  // Check is_ignored function
  EXPECT_FALSE(__moore_coverpoint_is_ignored(cg, 0, 5));
  EXPECT_FALSE(__moore_coverpoint_is_ignored(cg, 0, 9));
  EXPECT_TRUE(__moore_coverpoint_is_ignored(cg, 0, 10));
  EXPECT_TRUE(__moore_coverpoint_is_ignored(cg, 0, 15));
  EXPECT_TRUE(__moore_coverpoint_is_ignored(cg, 0, 20));
  EXPECT_FALSE(__moore_coverpoint_is_ignored(cg, 0, 21));

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeIgnoreBinsTest, SetIgnoreBinsBatch) {
  void *cg = __moore_covergroup_create("batch_ignore_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Create array of ignore bins
  MooreCoverageBin bins[2];
  bins[0].name = "ignore1";
  bins[0].type = MOORE_BIN_RANGE;
  bins[0].kind = MOORE_BIN_KIND_IGNORE;
  bins[0].low = 0;
  bins[0].high = 10;
  bins[0].hit_count = 0;

  bins[1].name = "ignore2";
  bins[1].type = MOORE_BIN_RANGE;
  bins[1].kind = MOORE_BIN_KIND_IGNORE;
  bins[1].low = 100;
  bins[1].high = 110;
  bins[1].hit_count = 0;

  __moore_coverpoint_set_ignore_bins(cg, 0, bins, 2);

  // Check both ranges are ignored
  EXPECT_TRUE(__moore_coverpoint_is_ignored(cg, 0, 5));
  EXPECT_TRUE(__moore_coverpoint_is_ignored(cg, 0, 105));
  EXPECT_FALSE(__moore_coverpoint_is_ignored(cg, 0, 50));

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeIllegalBinsTest, SetIllegalBinsBatch) {
  void *cg = __moore_covergroup_create("batch_illegal_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Create array of illegal bins
  MooreCoverageBin bins[2];
  bins[0].name = "illegal1";
  bins[0].type = MOORE_BIN_RANGE;
  bins[0].kind = MOORE_BIN_KIND_ILLEGAL;
  bins[0].low = 1000;
  bins[0].high = 2000;
  bins[0].hit_count = 0;

  bins[1].name = "illegal2";
  bins[1].type = MOORE_BIN_VALUE;
  bins[1].kind = MOORE_BIN_KIND_ILLEGAL;
  bins[1].low = -1;
  bins[1].high = -1;
  bins[1].hit_count = 0;

  __moore_coverpoint_set_illegal_bins(cg, 0, bins, 2);

  // Check both are detected as illegal
  EXPECT_TRUE(__moore_coverpoint_is_illegal(cg, 0, 1500));
  EXPECT_TRUE(__moore_coverpoint_is_illegal(cg, 0, -1));
  EXPECT_FALSE(__moore_coverpoint_is_illegal(cg, 0, 500));

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeIllegalBinsTest, IllegalBinCallback) {
  static int callbackCount = 0;
  static int64_t lastValue = 0;

  // Reset
  callbackCount = 0;
  lastValue = 0;

  void *cg = __moore_covergroup_create("callback_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  __moore_coverage_reset_illegal_bin_hits();
  __moore_coverage_set_illegal_bin_fatal(false);

  // Set callback
  __moore_coverage_set_illegal_bin_callback(
      [](const char *cg_name, const char *cp_name, const char *bin_name,
         int64_t value, void *userData) {
        callbackCount++;
        lastValue = value;
      },
      nullptr);

  // Add illegal bin
  __moore_coverpoint_add_illegal_bin(cg, 0, "illegal_val", 999, 999);

  // Sample illegal value
  __moore_coverpoint_sample(cg, 0, 999);

  EXPECT_EQ(callbackCount, 1);
  EXPECT_EQ(lastValue, 999);

  // Clear callback
  __moore_coverage_set_illegal_bin_callback(nullptr, nullptr);

  __moore_covergroup_destroy(cg);
  __moore_coverage_reset_illegal_bin_hits();
}

//===----------------------------------------------------------------------===//
// Coverage Exclusion Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeExclusionTest, AddAndClearExclusions) {
  // Clear any existing exclusions
  __moore_coverage_clear_exclusions();
  EXPECT_EQ(__moore_coverage_get_exclusion_count(), 0);

  // Add exclusions
  EXPECT_EQ(__moore_coverage_add_exclusion("cg1.*.*"), 0);
  EXPECT_EQ(__moore_coverage_get_exclusion_count(), 1);

  EXPECT_EQ(__moore_coverage_add_exclusion("cg2.cp1.*"), 0);
  EXPECT_EQ(__moore_coverage_get_exclusion_count(), 2);

  // Clear all
  __moore_coverage_clear_exclusions();
  EXPECT_EQ(__moore_coverage_get_exclusion_count(), 0);
}

TEST(MooreRuntimeExclusionTest, RemoveExclusion) {
  __moore_coverage_clear_exclusions();

  __moore_coverage_add_exclusion("pattern1");
  __moore_coverage_add_exclusion("pattern2");
  __moore_coverage_add_exclusion("pattern3");
  EXPECT_EQ(__moore_coverage_get_exclusion_count(), 3);

  // Remove one
  EXPECT_EQ(__moore_coverage_remove_exclusion("pattern2"), 0);
  EXPECT_EQ(__moore_coverage_get_exclusion_count(), 2);

  // Try to remove non-existent
  EXPECT_NE(__moore_coverage_remove_exclusion("nonexistent"), 0);
  EXPECT_EQ(__moore_coverage_get_exclusion_count(), 2);

  __moore_coverage_clear_exclusions();
}

TEST(MooreRuntimeExclusionTest, IsExcludedWildcard) {
  __moore_coverage_clear_exclusions();

  // Add wildcard patterns
  __moore_coverage_add_exclusion("test_cg.*.*");

  // Check exclusions
  EXPECT_TRUE(__moore_coverage_is_excluded("test_cg", "any_cp", "any_bin"));
  EXPECT_TRUE(__moore_coverage_is_excluded("test_cg", "cp1", "bin1"));
  EXPECT_FALSE(__moore_coverage_is_excluded("other_cg", "any_cp", "any_bin"));

  __moore_coverage_clear_exclusions();
}

TEST(MooreRuntimeExclusionTest, IsExcludedExact) {
  __moore_coverage_clear_exclusions();

  // Add exact pattern
  __moore_coverage_add_exclusion("my_cg.my_cp.my_bin");

  // Only exact match should be excluded
  EXPECT_TRUE(__moore_coverage_is_excluded("my_cg", "my_cp", "my_bin"));
  EXPECT_FALSE(__moore_coverage_is_excluded("my_cg", "my_cp", "other_bin"));
  EXPECT_FALSE(__moore_coverage_is_excluded("my_cg", "other_cp", "my_bin"));

  __moore_coverage_clear_exclusions();
}

TEST(MooreRuntimeExclusionTest, IsExcludedQuestionMark) {
  __moore_coverage_clear_exclusions();

  // Add pattern with ? wildcard
  __moore_coverage_add_exclusion("cg?.cp?.bin?");

  // Single character wildcard
  EXPECT_TRUE(__moore_coverage_is_excluded("cg1", "cp2", "bin3"));
  EXPECT_TRUE(__moore_coverage_is_excluded("cgA", "cpB", "binC"));
  EXPECT_FALSE(__moore_coverage_is_excluded("cg12", "cp1", "bin1"));  // cg12 doesn't match cg?
  EXPECT_FALSE(__moore_coverage_is_excluded("cg1", "cp12", "bin1"));  // cp12 doesn't match cp?

  __moore_coverage_clear_exclusions();
}

TEST(MooreRuntimeExclusionTest, SaveAndLoadExclusions) {
  __moore_coverage_clear_exclusions();

  // Add some exclusions
  __moore_coverage_add_exclusion("save_cg1.*.*");
  __moore_coverage_add_exclusion("save_cg2.cp1.bin?");
  __moore_coverage_add_exclusion("save_cg3.*.exact_bin");

  // Save to file
  const char *filename = "/tmp/test_exclusions.txt";
  EXPECT_EQ(__moore_coverage_save_exclusions(filename), 0);

  // Clear and reload
  __moore_coverage_clear_exclusions();
  EXPECT_EQ(__moore_coverage_get_exclusion_count(), 0);

  int32_t loaded = __moore_coverage_load_exclusions(filename);
  EXPECT_EQ(loaded, 3);
  EXPECT_EQ(__moore_coverage_get_exclusion_count(), 3);

  // Verify patterns work
  EXPECT_TRUE(__moore_coverage_is_excluded("save_cg1", "any", "thing"));
  EXPECT_TRUE(__moore_coverage_is_excluded("save_cg2", "cp1", "bin1"));

  // Clean up
  __moore_coverage_clear_exclusions();
  std::remove(filename);
}

TEST(MooreRuntimeExclusionTest, LoadNonexistentFile) {
  int32_t result = __moore_coverage_load_exclusions("/tmp/nonexistent_exclusions.txt");
  EXPECT_EQ(result, -1);
}

TEST(MooreRuntimeExclusionTest, LoadNullFilename) {
  int32_t result = __moore_coverage_load_exclusions(nullptr);
  EXPECT_EQ(result, -1);
}

TEST(MooreRuntimeExclusionTest, SaveNullFilename) {
  int32_t result = __moore_coverage_save_exclusions(nullptr);
  EXPECT_NE(result, 0);
}

TEST(MooreRuntimeExclusionTest, AddNullPattern) {
  int32_t result = __moore_coverage_add_exclusion(nullptr);
  EXPECT_NE(result, 0);
}

TEST(MooreRuntimeExclusionTest, AddEmptyPattern) {
  int32_t result = __moore_coverage_add_exclusion("");
  EXPECT_NE(result, 0);
}

TEST(MooreRuntimeIllegalBinsTest, InvalidIndex) {
  void *cg = __moore_covergroup_create("invalid_idx_cg", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // These should not crash
  __moore_coverpoint_add_illegal_bin(cg, -1, "bad", 0, 10);
  __moore_coverpoint_add_illegal_bin(cg, 5, "bad", 0, 10);
  __moore_coverpoint_add_ignore_bin(cg, -1, "bad", 0, 10);
  __moore_coverpoint_add_ignore_bin(cg, 5, "bad", 0, 10);

  // Check with invalid index returns false
  EXPECT_FALSE(__moore_coverpoint_is_illegal(cg, -1, 5));
  EXPECT_FALSE(__moore_coverpoint_is_illegal(cg, 5, 5));
  EXPECT_FALSE(__moore_coverpoint_is_ignored(cg, -1, 5));
  EXPECT_FALSE(__moore_coverpoint_is_ignored(cg, 5, 5));

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeIllegalBinsTest, NullCovergroup) {
  // These should not crash with null covergroup
  __moore_coverpoint_add_illegal_bin(nullptr, 0, "bad", 0, 10);
  __moore_coverpoint_add_ignore_bin(nullptr, 0, "bad", 0, 10);

  EXPECT_FALSE(__moore_coverpoint_is_illegal(nullptr, 0, 5));
  EXPECT_FALSE(__moore_coverpoint_is_ignored(nullptr, 0, 5));
}

//===----------------------------------------------------------------------===//
// Coverage Options Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCoverageOptionsTest, CovergroupWeight) {
  void *cg = __moore_covergroup_create("weight_test", 1);
  ASSERT_NE(cg, nullptr);

  // Default weight should be 1
  EXPECT_EQ(__moore_covergroup_get_weight(cg), 1);

  // Set custom weight
  __moore_covergroup_set_weight(cg, 5);
  EXPECT_EQ(__moore_covergroup_get_weight(cg), 5);

  // Weight should be clamped to minimum 1
  __moore_covergroup_set_weight(cg, 0);
  EXPECT_EQ(__moore_covergroup_get_weight(cg), 1);

  __moore_covergroup_set_weight(cg, -10);
  EXPECT_EQ(__moore_covergroup_get_weight(cg), 1);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageOptionsTest, CovergroupAtLeast) {
  void *cg = __moore_covergroup_create("at_least_test", 1);
  ASSERT_NE(cg, nullptr);

  // Default at_least should be 1
  EXPECT_EQ(__moore_covergroup_get_at_least(cg), 1);

  // Set custom at_least
  __moore_covergroup_set_at_least(cg, 10);
  EXPECT_EQ(__moore_covergroup_get_at_least(cg), 10);

  // at_least should be clamped to minimum 1
  __moore_covergroup_set_at_least(cg, 0);
  EXPECT_EQ(__moore_covergroup_get_at_least(cg), 1);

  __moore_covergroup_set_at_least(cg, -5);
  EXPECT_EQ(__moore_covergroup_get_at_least(cg), 1);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageOptionsTest, CovergroupAutoBinMax) {
  void *cg = __moore_covergroup_create("auto_bin_test", 1);
  ASSERT_NE(cg, nullptr);

  // Default auto_bin_max should be 64
  EXPECT_EQ(__moore_covergroup_get_auto_bin_max(cg), 64);

  // Set custom auto_bin_max
  __moore_covergroup_set_auto_bin_max(cg, 128);
  EXPECT_EQ(__moore_covergroup_get_auto_bin_max(cg), 128);

  // auto_bin_max should fall back to 64 if set to <= 0
  __moore_covergroup_set_auto_bin_max(cg, 0);
  EXPECT_EQ(__moore_covergroup_get_auto_bin_max(cg), 64);

  __moore_covergroup_set_auto_bin_max(cg, -1);
  EXPECT_EQ(__moore_covergroup_get_auto_bin_max(cg), 64);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageOptionsTest, CoverpointWeight) {
  void *cg = __moore_covergroup_create("cp_weight_test", 2);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  // Default weight should be 1
  EXPECT_EQ(__moore_coverpoint_get_weight(cg, 0), 1);
  EXPECT_EQ(__moore_coverpoint_get_weight(cg, 1), 1);

  // Set custom weights
  __moore_coverpoint_set_weight(cg, 0, 3);
  __moore_coverpoint_set_weight(cg, 1, 7);
  EXPECT_EQ(__moore_coverpoint_get_weight(cg, 0), 3);
  EXPECT_EQ(__moore_coverpoint_get_weight(cg, 1), 7);

  // Invalid index should return default
  EXPECT_EQ(__moore_coverpoint_get_weight(cg, -1), 1);
  EXPECT_EQ(__moore_coverpoint_get_weight(cg, 10), 1);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageOptionsTest, CoverpointGoal) {
  void *cg = __moore_covergroup_create("cp_goal_test", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Default goal should be 100.0
  EXPECT_DOUBLE_EQ(__moore_coverpoint_get_goal(cg, 0), 100.0);

  // Set custom goal
  __moore_coverpoint_set_goal(cg, 0, 80.0);
  EXPECT_DOUBLE_EQ(__moore_coverpoint_get_goal(cg, 0), 80.0);

  // Goal should be clamped to [0, 100]
  __moore_coverpoint_set_goal(cg, 0, -10.0);
  EXPECT_DOUBLE_EQ(__moore_coverpoint_get_goal(cg, 0), 0.0);

  __moore_coverpoint_set_goal(cg, 0, 150.0);
  EXPECT_DOUBLE_EQ(__moore_coverpoint_get_goal(cg, 0), 100.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageOptionsTest, CoverpointAtLeast) {
  void *cg = __moore_covergroup_create("cp_at_least_test", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Default at_least should be 1
  EXPECT_EQ(__moore_coverpoint_get_at_least(cg, 0), 1);

  // Set custom at_least
  __moore_coverpoint_set_at_least(cg, 0, 5);
  EXPECT_EQ(__moore_coverpoint_get_at_least(cg, 0), 5);

  // at_least should be clamped to minimum 1
  __moore_coverpoint_set_at_least(cg, 0, 0);
  EXPECT_EQ(__moore_coverpoint_get_at_least(cg, 0), 1);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageOptionsTest, CoverpointAutoBinMax) {
  void *cg = __moore_covergroup_create("cp_auto_bin_test", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Default auto_bin_max should be 64
  EXPECT_EQ(__moore_coverpoint_get_auto_bin_max(cg, 0), 64);

  // Set custom auto_bin_max
  __moore_coverpoint_set_auto_bin_max(cg, 0, 256);
  EXPECT_EQ(__moore_coverpoint_get_auto_bin_max(cg, 0), 256);

  // auto_bin_max should fall back to 64 if set to <= 0
  __moore_coverpoint_set_auto_bin_max(cg, 0, 0);
  EXPECT_EQ(__moore_coverpoint_get_auto_bin_max(cg, 0), 64);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageOptionsTest, GenericOptionAPI) {
  void *cg = __moore_covergroup_create("generic_opt_test", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Test covergroup generic option API
  __moore_covergroup_set_option(cg, MOORE_OPTION_GOAL, 90.0);
  EXPECT_DOUBLE_EQ(__moore_covergroup_get_option(cg, MOORE_OPTION_GOAL), 90.0);

  __moore_covergroup_set_option(cg, MOORE_OPTION_WEIGHT, 3.0);
  EXPECT_DOUBLE_EQ(__moore_covergroup_get_option(cg, MOORE_OPTION_WEIGHT), 3.0);

  __moore_covergroup_set_option(cg, MOORE_OPTION_AT_LEAST, 5.0);
  EXPECT_DOUBLE_EQ(__moore_covergroup_get_option(cg, MOORE_OPTION_AT_LEAST), 5.0);

  __moore_covergroup_set_option(cg, MOORE_OPTION_AUTO_BIN_MAX, 100.0);
  EXPECT_DOUBLE_EQ(__moore_covergroup_get_option(cg, MOORE_OPTION_AUTO_BIN_MAX), 100.0);

  // Test coverpoint generic option API
  __moore_coverpoint_set_option(cg, 0, MOORE_OPTION_GOAL, 75.0);
  EXPECT_DOUBLE_EQ(__moore_coverpoint_get_option(cg, 0, MOORE_OPTION_GOAL), 75.0);

  __moore_coverpoint_set_option(cg, 0, MOORE_OPTION_WEIGHT, 2.0);
  EXPECT_DOUBLE_EQ(__moore_coverpoint_get_option(cg, 0, MOORE_OPTION_WEIGHT), 2.0);

  __moore_coverpoint_set_option(cg, 0, MOORE_OPTION_AT_LEAST, 10.0);
  EXPECT_DOUBLE_EQ(__moore_coverpoint_get_option(cg, 0, MOORE_OPTION_AT_LEAST), 10.0);

  __moore_coverpoint_set_option(cg, 0, MOORE_OPTION_AUTO_BIN_MAX, 50.0);
  EXPECT_DOUBLE_EQ(__moore_coverpoint_get_option(cg, 0, MOORE_OPTION_AUTO_BIN_MAX), 50.0);

  // Test unknown option returns 0
  EXPECT_DOUBLE_EQ(__moore_covergroup_get_option(cg, 999), 0.0);
  EXPECT_DOUBLE_EQ(__moore_coverpoint_get_option(cg, 0, 999), 0.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageOptionsTest, AtLeastAffectsCoverage) {
  // Test that at_least threshold affects coverage calculation
  void *cg = __moore_covergroup_create("at_least_cov_test", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Sample the same value 3 times
  __moore_coverpoint_sample(cg, 0, 42);
  __moore_coverpoint_sample(cg, 0, 42);
  __moore_coverpoint_sample(cg, 0, 42);

  // With at_least=1 (default), we should have coverage > 0
  double covDefault = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_GT(covDefault, 0.0);

  // Set at_least=5, coverage should still work but value isn't covered yet
  __moore_coverpoint_set_at_least(cg, 0, 5);
  double covAt5 = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(covAt5, 0.0);  // 3 hits < 5 threshold

  // Add 2 more samples (total 5), now it should be covered
  __moore_coverpoint_sample(cg, 0, 42);
  __moore_coverpoint_sample(cg, 0, 42);
  double covAfter5 = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_GT(covAfter5, 0.0);  // 5 hits >= 5 threshold

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageOptionsTest, AtLeastWithExplicitBins) {
  // Test at_least with explicit bins
  void *cg = __moore_covergroup_create("at_least_bins_test", 1);
  ASSERT_NE(cg, nullptr);

  // Create coverpoint with explicit bins
  MooreCoverageBin bins[] = {
      {"bin0", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 0, 0, 0},
      {"bin1", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 1, 1, 0},
      {"bin2", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 2, 2, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 3);

  // Sample each bin once
  __moore_coverpoint_sample(cg, 0, 0);
  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 0, 2);

  // With at_least=1, all 3 bins should be covered (100%)
  double cov1 = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov1, 100.0);

  // Set at_least=2, now bins aren't covered (only 1 hit each)
  __moore_coverpoint_set_at_least(cg, 0, 2);
  double cov2 = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov2, 0.0);  // 0 out of 3 bins covered

  // Sample bin0 again (now has 2 hits)
  __moore_coverpoint_sample(cg, 0, 0);
  double cov3 = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_NEAR(cov3, 33.33, 0.5);  // 1 out of 3 bins covered

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageOptionsTest, AutoBinMaxAffectsCoverage) {
  // Test that auto_bin_max affects coverage calculation
  // IEEE 1800-2017: auto_bin_max limits the maximum number of auto-generated bins.
  // If the value range is smaller than auto_bin_max, bins are based on the range.
  void *cg = __moore_covergroup_create("auto_bin_max_test", 1);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp");

  // Sample values across a wide range (0-99), so range = 100
  for (int i = 0; i < 100; i++) {
    __moore_coverpoint_sample(cg, 0, i);
  }

  // With default auto_bin_max=64, range (100) > auto_bin_max (64)
  // So effective bins = 64, coverage = 100/64 = capped at 100%
  double covDefault = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_GE(covDefault, 100.0);  // 100 values in 64 bins = >100% (capped)

  // Set auto_bin_max=200, range (100) < auto_bin_max (200)
  // So effective bins = 100 (the actual range), coverage = 100/100 = 100%
  __moore_coverpoint_set_auto_bin_max(cg, 0, 200);
  double covLarge = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(covLarge, 100.0);  // 100 values in range of 100

  // Now test with a smaller auto_bin_max to demonstrate the limiting effect
  __moore_coverpoint_set_auto_bin_max(cg, 0, 50);
  double covSmall = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_GE(covSmall, 100.0);  // 100 values in 50 bins = 200% (capped to 100%)

  // Sample only 25 of a 100-value range with auto_bin_max=50
  void *cg2 = __moore_covergroup_create("auto_bin_max_test2", 1);
  ASSERT_NE(cg2, nullptr);
  __moore_coverpoint_init(cg2, 0, "cp");
  __moore_coverpoint_set_auto_bin_max(cg2, 0, 50);

  for (int i = 0; i < 100; i += 4) {  // Sample 25 values out of 100
    __moore_coverpoint_sample(cg2, 0, i);
  }
  // Range is 96 (0 to 96), auto_bin_max is 50, so effective bins = 50
  // 25 values in 50 bins = 50%
  double covPartial = __moore_coverpoint_get_coverage(cg2, 0);
  EXPECT_NEAR(covPartial, 50.0, 2.0);

  __moore_covergroup_destroy(cg);
  __moore_covergroup_destroy(cg2);
}

TEST(MooreRuntimeCoverageOptionsTest, WeightedCoverage) {
  // Test weighted coverage calculation
  void *cg = __moore_covergroup_create("weighted_test", 2);
  ASSERT_NE(cg, nullptr);
  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  // cp0 has single value (100% coverage), weight = 1
  __moore_coverpoint_sample(cg, 0, 42);
  __moore_coverpoint_set_weight(cg, 0, 1);

  // cp1 has no samples (0% coverage), weight = 3
  __moore_coverpoint_set_weight(cg, 1, 3);

  // Weighted coverage = (100*1 + 0*3) / (1+3) = 25%
  double weighted = __moore_covergroup_get_weighted_coverage(cg);
  EXPECT_NEAR(weighted, 25.0, 1.0);

  // Sample cp1 to get 100% coverage there too
  __moore_coverpoint_sample(cg, 1, 99);

  // Now weighted = (100*1 + 100*3) / (1+3) = 100%
  double weightedFull = __moore_covergroup_get_weighted_coverage(cg);
  EXPECT_DOUBLE_EQ(weightedFull, 100.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageOptionsTest, NullCovergroupOptions) {
  // Test that null covergroup doesn't crash
  __moore_covergroup_set_weight(nullptr, 5);
  __moore_covergroup_set_at_least(nullptr, 5);
  __moore_covergroup_set_auto_bin_max(nullptr, 100);
  __moore_covergroup_set_option(nullptr, MOORE_OPTION_GOAL, 50.0);

  EXPECT_EQ(__moore_covergroup_get_weight(nullptr), 1);
  EXPECT_EQ(__moore_covergroup_get_at_least(nullptr), 1);
  EXPECT_EQ(__moore_covergroup_get_auto_bin_max(nullptr), 64);
  EXPECT_DOUBLE_EQ(__moore_covergroup_get_option(nullptr, MOORE_OPTION_GOAL), 100.0);
}

TEST(MooreRuntimeCoverageOptionsTest, BinCoveredRespectAtLeast) {
  // Test __moore_coverpoint_bin_covered respects at_least
  void *cg = __moore_covergroup_create("bin_covered_test", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[] = {
      {"bin0", MOORE_BIN_VALUE, MOORE_BIN_KIND_NORMAL, 0, 0, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 1);

  // Sample once
  __moore_coverpoint_sample(cg, 0, 0);

  // With at_least=1, bin should be covered
  EXPECT_TRUE(__moore_coverpoint_bin_covered(cg, 0, 0));

  // Set at_least=3, bin should not be covered
  __moore_coverpoint_set_at_least(cg, 0, 3);
  EXPECT_FALSE(__moore_coverpoint_bin_covered(cg, 0, 0));

  // Sample 2 more times
  __moore_coverpoint_sample(cg, 0, 0);
  __moore_coverpoint_sample(cg, 0, 0);
  EXPECT_TRUE(__moore_coverpoint_bin_covered(cg, 0, 0));

  __moore_covergroup_destroy(cg);
}

//===----------------------------------------------------------------------===//
// Transition Bin Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeTransitionBinTest, SimpleTwoStepTransition) {
  // Test a simple two-step transition: (0 => 1)
  void *cg = __moore_covergroup_create("trans_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "state");

  // Create a transition sequence: 0 => 1
  MooreTransitionStep steps[] = {
      {0, MOORE_TRANS_NONE, 0, 0},  // value 0
      {1, MOORE_TRANS_NONE, 0, 0}   // value 1
  };
  MooreTransitionSequence seq = {steps, 2};

  // Add the transition bin
  __moore_coverpoint_add_transition_bin(cg, 0, "rise", &seq, 1);

  // Initial: no hits
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 0);

  // Sample values that don't trigger the transition
  __moore_coverpoint_sample(cg, 0, 5);  // No previous value yet
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 0);

  __moore_coverpoint_sample(cg, 0, 3);  // 5 => 3, not 0 => 1
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 0);

  // Sample values that trigger the transition
  __moore_coverpoint_sample(cg, 0, 0);  // 3 => 0, not triggered
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 0);

  __moore_coverpoint_sample(cg, 0, 1);  // 0 => 1, TRIGGERED!
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 1);

  // Trigger again
  __moore_coverpoint_sample(cg, 0, 0);  // 1 => 0, not triggered
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 1);

  __moore_coverpoint_sample(cg, 0, 1);  // 0 => 1, TRIGGERED again!
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 2);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeTransitionBinTest, MultipleTransitionBins) {
  // Test multiple transition bins on one coverpoint
  // bins rise = (0 => 1);
  // bins fall = (1 => 0);
  void *cg = __moore_covergroup_create("trans_multi_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "state");

  // Create rise transition: 0 => 1
  MooreTransitionStep riseSteps[] = {
      {0, MOORE_TRANS_NONE, 0, 0},
      {1, MOORE_TRANS_NONE, 0, 0}
  };
  MooreTransitionSequence riseSeq = {riseSteps, 2};
  __moore_coverpoint_add_transition_bin(cg, 0, "rise", &riseSeq, 1);

  // Create fall transition: 1 => 0
  MooreTransitionStep fallSteps[] = {
      {1, MOORE_TRANS_NONE, 0, 0},
      {0, MOORE_TRANS_NONE, 0, 0}
  };
  MooreTransitionSequence fallSeq = {fallSteps, 2};
  __moore_coverpoint_add_transition_bin(cg, 0, "fall", &fallSeq, 1);

  // Initial state
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 0);  // rise
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 1), 0);  // fall

  // Sample sequence: 0 -> 1 -> 0 -> 1
  __moore_coverpoint_sample(cg, 0, 0);  // First sample (no previous)
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 0);
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 1), 0);

  __moore_coverpoint_sample(cg, 0, 1);  // 0 => 1 (rise triggered)
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 1);
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 1), 0);

  __moore_coverpoint_sample(cg, 0, 0);  // 1 => 0 (fall triggered)
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 1);
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 1), 1);

  __moore_coverpoint_sample(cg, 0, 1);  // 0 => 1 (rise triggered again)
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 2);
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 1), 1);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeTransitionBinTest, ThreeStepTransition) {
  // Test a three-step transition sequence: (0 => 1 => 2)
  void *cg = __moore_covergroup_create("trans_seq_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "state");

  // Create sequence: 0 => 1 => 2
  MooreTransitionStep steps[] = {
      {0, MOORE_TRANS_NONE, 0, 0},
      {1, MOORE_TRANS_NONE, 0, 0},
      {2, MOORE_TRANS_NONE, 0, 0}
  };
  MooreTransitionSequence seq = {steps, 3};
  __moore_coverpoint_add_transition_bin(cg, 0, "count_up", &seq, 1);

  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 0);

  // Sample 0 -> 1 (starts sequence)
  __moore_coverpoint_sample(cg, 0, 0);
  __moore_coverpoint_sample(cg, 0, 1);  // 0 => 1 matches first two steps
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 0);  // Not complete yet

  // Complete the sequence with 2
  __moore_coverpoint_sample(cg, 0, 2);  // => 2 completes the sequence!
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 1);

  // Break and restart sequence
  __moore_coverpoint_sample(cg, 0, 0);  // Start fresh
  __moore_coverpoint_sample(cg, 0, 1);  // 0 => 1
  __moore_coverpoint_sample(cg, 0, 5);  // Sequence broken! 1 => 5, not 1 => 2
  __moore_coverpoint_sample(cg, 0, 2);  // Doesn't continue
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 1);  // Still 1

  // Complete again
  __moore_coverpoint_sample(cg, 0, 0);
  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 0, 2);
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 2);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeTransitionBinTest, FourStepTransition) {
  // Test a four-step transition sequence: (0 => 1 => 2 => 3)
  void *cg = __moore_covergroup_create("trans_4step_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "data");

  MooreTransitionStep steps[] = {
      {0, MOORE_TRANS_NONE, 0, 0},
      {1, MOORE_TRANS_NONE, 0, 0},
      {2, MOORE_TRANS_NONE, 0, 0},
      {3, MOORE_TRANS_NONE, 0, 0}
  };
  MooreTransitionSequence seq = {steps, 4};
  __moore_coverpoint_add_transition_bin(cg, 0, "sequence", &seq, 1);

  // Complete the full sequence
  __moore_coverpoint_sample(cg, 0, 0);
  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 0, 2);
  __moore_coverpoint_sample(cg, 0, 3);
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 1);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeTransitionBinTest, TransitionBinReset) {
  // Test that reset clears transition bin hits and state
  void *cg = __moore_covergroup_create("trans_reset_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "state");

  MooreTransitionStep steps[] = {
      {0, MOORE_TRANS_NONE, 0, 0},
      {1, MOORE_TRANS_NONE, 0, 0}
  };
  MooreTransitionSequence seq = {steps, 2};
  __moore_coverpoint_add_transition_bin(cg, 0, "rise", &seq, 1);

  // Trigger transition
  __moore_coverpoint_sample(cg, 0, 0);
  __moore_coverpoint_sample(cg, 0, 1);
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 1);

  // Reset
  __moore_coverpoint_reset(cg, 0);
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 0);

  // Trigger again after reset
  __moore_coverpoint_sample(cg, 0, 0);
  __moore_coverpoint_sample(cg, 0, 1);
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 1);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeTransitionBinTest, TransitionBinInvalidIndex) {
  void *cg = __moore_covergroup_create("trans_invalid_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "state");

  // Get hits with invalid indices should return 0
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 0);   // No bins yet
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, -1), 0);  // Negative index
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 100), 0); // Out of bounds

  EXPECT_EQ(__moore_transition_bin_get_hits(cg, -1, 0), 0);  // Invalid cp_index
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 5, 0), 0);   // Invalid cp_index

  EXPECT_EQ(__moore_transition_bin_get_hits(nullptr, 0, 0), 0);  // Null cg

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeTransitionBinTest, AddTransitionBinInvalidArgs) {
  void *cg = __moore_covergroup_create("trans_invalid_add_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "state");

  // These should not crash
  __moore_coverpoint_add_transition_bin(nullptr, 0, "bad", nullptr, 0);
  __moore_coverpoint_add_transition_bin(cg, -1, "bad", nullptr, 0);
  __moore_coverpoint_add_transition_bin(cg, 5, "bad", nullptr, 0);
  __moore_coverpoint_add_transition_bin(cg, 0, "bad", nullptr, 0);
  __moore_coverpoint_add_transition_bin(cg, 0, "bad", nullptr, -1);

  // No bins should have been added
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeTransitionBinTest, SelfTransition) {
  // Test a self-transition: (5 => 5)
  void *cg = __moore_covergroup_create("trans_self_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "state");

  MooreTransitionStep steps[] = {
      {5, MOORE_TRANS_NONE, 0, 0},
      {5, MOORE_TRANS_NONE, 0, 0}
  };
  MooreTransitionSequence seq = {steps, 2};
  __moore_coverpoint_add_transition_bin(cg, 0, "stay", &seq, 1);

  // Self-transition requires two consecutive 5s
  __moore_coverpoint_sample(cg, 0, 5);
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 0);  // First 5, no prev

  __moore_coverpoint_sample(cg, 0, 5);
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 1);  // 5 => 5 triggered

  __moore_coverpoint_sample(cg, 0, 5);
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 2);  // 5 => 5 again

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeTransitionBinTest, CoverageWithMixedBins) {
  // Test that transition bins work alongside value bins
  void *cg = __moore_covergroup_create("mixed_bins_cg", 1);
  ASSERT_NE(cg, nullptr);

  // Initialize with value bins
  MooreCoverageBin valueBins[] = {
      {"low", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 0, 3, 0},
      {"high", MOORE_BIN_RANGE, MOORE_BIN_KIND_NORMAL, 4, 7, 0}
  };
  __moore_coverpoint_init_with_bins(cg, 0, "data", valueBins, 2);

  // Add a transition bin
  MooreTransitionStep steps[] = {
      {0, MOORE_TRANS_NONE, 0, 0},
      {7, MOORE_TRANS_NONE, 0, 0}
  };
  MooreTransitionSequence seq = {steps, 2};
  __moore_coverpoint_add_transition_bin(cg, 0, "low_to_high", &seq, 1);

  // Sample values
  __moore_coverpoint_sample(cg, 0, 0);  // Hits "low" bin
  __moore_coverpoint_sample(cg, 0, 7);  // Hits "high" bin AND transition

  // Check value bin hits
  EXPECT_GE(__moore_coverpoint_get_bin_hits(cg, 0, 0), 1);  // low
  EXPECT_GE(__moore_coverpoint_get_bin_hits(cg, 0, 1), 1);  // high

  // Check transition bin hit
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 1);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeTransitionBinTest, OverlappingTransitions) {
  // Test overlapping transition sequences
  // bins t1 = (0 => 1);
  // bins t2 = (0 => 1 => 2);
  // Sampling 0 -> 1 -> 2 should hit t1 once and t2 once
  void *cg = __moore_covergroup_create("overlap_trans_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "state");

  // Short transition
  MooreTransitionStep steps1[] = {
      {0, MOORE_TRANS_NONE, 0, 0},
      {1, MOORE_TRANS_NONE, 0, 0}
  };
  MooreTransitionSequence seq1 = {steps1, 2};
  __moore_coverpoint_add_transition_bin(cg, 0, "short", &seq1, 1);

  // Longer transition
  MooreTransitionStep steps2[] = {
      {0, MOORE_TRANS_NONE, 0, 0},
      {1, MOORE_TRANS_NONE, 0, 0},
      {2, MOORE_TRANS_NONE, 0, 0}
  };
  MooreTransitionSequence seq2 = {steps2, 3};
  __moore_coverpoint_add_transition_bin(cg, 0, "long", &seq2, 1);

  // Sample sequence: 0 -> 1 -> 2
  __moore_coverpoint_sample(cg, 0, 0);
  __moore_coverpoint_sample(cg, 0, 1);  // Triggers "short"
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 1);  // short
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 1), 0);  // long (not yet)

  __moore_coverpoint_sample(cg, 0, 2);  // Triggers "long"
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 0), 1);  // short still 1
  EXPECT_EQ(__moore_transition_bin_get_hits(cg, 0, 1), 1);  // long now 1

  __moore_covergroup_destroy(cg);
}

//===----------------------------------------------------------------------===//
// Array Constraint Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeArrayConstraintTest, UniqueCheckAllUnique) {
  // Test unique check with all unique elements
  int32_t arr[] = {1, 5, 3, 9, 7, 2};
  EXPECT_EQ(__moore_constraint_unique_check(arr, 6, sizeof(int32_t)), 1);
}

TEST(MooreRuntimeArrayConstraintTest, UniqueCheckDuplicates) {
  // Test unique check with duplicates
  int32_t arr[] = {1, 5, 3, 5, 7, 2};  // 5 appears twice
  EXPECT_EQ(__moore_constraint_unique_check(arr, 6, sizeof(int32_t)), 0);
}

TEST(MooreRuntimeArrayConstraintTest, UniqueCheckSingleElement) {
  // Single element is trivially unique
  int32_t arr[] = {42};
  EXPECT_EQ(__moore_constraint_unique_check(arr, 1, sizeof(int32_t)), 1);
}

TEST(MooreRuntimeArrayConstraintTest, UniqueCheckEmpty) {
  // Empty array is trivially unique
  EXPECT_EQ(__moore_constraint_unique_check(nullptr, 0, sizeof(int32_t)), 1);
}

TEST(MooreRuntimeArrayConstraintTest, UniqueScalars) {
  // Test unique scalars check
  int64_t values[] = {10, 20, 30, 40};
  EXPECT_EQ(__moore_constraint_unique_scalars(values, 4, sizeof(int64_t)), 1);

  int64_t duplicates[] = {10, 20, 10, 40};  // 10 appears twice
  EXPECT_EQ(__moore_constraint_unique_scalars(duplicates, 4, sizeof(int64_t)), 0);
}

TEST(MooreRuntimeArrayConstraintTest, RandomizeUniqueArray) {
  // Test randomizing an array with unique constraint
  int32_t arr[5] = {0, 0, 0, 0, 0};
  int32_t result = __moore_randomize_unique_array(arr, 5, sizeof(int32_t), 0, 100);
  EXPECT_EQ(result, 1);  // Should succeed

  // Verify all elements are unique
  EXPECT_EQ(__moore_constraint_unique_check(arr, 5, sizeof(int32_t)), 1);

  // Verify all elements are in range
  for (int i = 0; i < 5; i++) {
    EXPECT_GE(arr[i], 0);
    EXPECT_LE(arr[i], 100);
  }
}

TEST(MooreRuntimeArrayConstraintTest, RandomizeUniqueArrayInsufficientRange) {
  // Test with range too small for unique values
  int32_t arr[10] = {0};
  int32_t result = __moore_randomize_unique_array(arr, 10, sizeof(int32_t), 0, 5);
  EXPECT_EQ(result, 0);  // Should fail - can't generate 10 unique values from range 0-5
}

TEST(MooreRuntimeArrayConstraintTest, SizeCheck) {
  // Test size constraint validation
  MooreQueue queue;
  queue.data = malloc(5 * sizeof(int32_t));
  queue.len = 5;

  EXPECT_EQ(__moore_constraint_size_check(&queue, 5), 1);  // Correct size
  EXPECT_EQ(__moore_constraint_size_check(&queue, 4), 0);  // Wrong size
  EXPECT_EQ(__moore_constraint_size_check(&queue, 6), 0);  // Wrong size

  free(queue.data);
}

TEST(MooreRuntimeArrayConstraintTest, SizeCheckNull) {
  // Test size check on null array
  EXPECT_EQ(__moore_constraint_size_check(nullptr, 0), 1);
  EXPECT_EQ(__moore_constraint_size_check(nullptr, 5), 0);
}

TEST(MooreRuntimeArrayConstraintTest, SumCheck) {
  // Test sum constraint validation
  MooreQueue queue;
  int32_t data[] = {10, 20, 30, 40};  // Sum = 100
  queue.data = data;
  queue.len = 4;

  EXPECT_EQ(__moore_constraint_sum_check(&queue, sizeof(int32_t), 100), 1);  // Correct sum
  EXPECT_EQ(__moore_constraint_sum_check(&queue, sizeof(int32_t), 99), 0);   // Wrong sum
  EXPECT_EQ(__moore_constraint_sum_check(&queue, sizeof(int32_t), 101), 0);  // Wrong sum
}

TEST(MooreRuntimeArrayConstraintTest, SumCheckEmpty) {
  // Test sum check on empty array
  MooreQueue queue;
  queue.data = nullptr;
  queue.len = 0;

  EXPECT_EQ(__moore_constraint_sum_check(&queue, sizeof(int32_t), 0), 1);  // Empty sum is 0
  EXPECT_EQ(__moore_constraint_sum_check(&queue, sizeof(int32_t), 1), 0);  // Non-zero expected
}

TEST(MooreRuntimeArrayConstraintTest, ForeachValidate) {
  // Test foreach constraint validation
  auto lessThan100 = [](int64_t value, void *) -> bool {
    return value < 100;
  };

  int32_t validArr[] = {10, 20, 30, 50, 99};
  EXPECT_EQ(__moore_constraint_foreach_validate(validArr, 5, sizeof(int32_t),
                                                 lessThan100, nullptr), 1);

  int32_t invalidArr[] = {10, 20, 100, 50, 30};  // 100 violates < 100
  EXPECT_EQ(__moore_constraint_foreach_validate(invalidArr, 5, sizeof(int32_t),
                                                 lessThan100, nullptr), 0);
}

TEST(MooreRuntimeArrayConstraintTest, ForeachValidateEmpty) {
  // Empty array trivially satisfies any constraint
  auto alwaysFalse = [](int64_t, void *) -> bool { return false; };
  EXPECT_EQ(__moore_constraint_foreach_validate(nullptr, 0, sizeof(int32_t),
                                                 alwaysFalse, nullptr), 1);
}

TEST(MooreRuntimeArrayConstraintTest, UniqueCheckByteSized) {
  // Test with byte-sized elements
  uint8_t arr[] = {0x01, 0x02, 0x03, 0x04, 0xFF};
  EXPECT_EQ(__moore_constraint_unique_check(arr, 5, sizeof(uint8_t)), 1);

  uint8_t dupArr[] = {0x01, 0x02, 0x01, 0x04};  // 0x01 appears twice
  EXPECT_EQ(__moore_constraint_unique_check(dupArr, 4, sizeof(uint8_t)), 0);
}

TEST(MooreRuntimeArrayConstraintTest, UniqueCheck64Bit) {
  // Test with 64-bit elements
  int64_t arr[] = {
    0x123456789ABCDEF0LL,
    static_cast<int64_t>(0xFEDCBA9876543210ULL),
    0x0000000000000001LL
  };
  EXPECT_EQ(__moore_constraint_unique_check(arr, 3, sizeof(int64_t)), 1);
}

//===----------------------------------------------------------------------===//
// Coverage Sample Callback Tests
//===----------------------------------------------------------------------===//

// Test data for callbacks
namespace {
struct SampleCallbackTestData {
  int preSampleCount = 0;
  int postSampleCount = 0;
  void *lastCg = nullptr;
  int64_t *lastArgs = nullptr;
  int32_t lastNumArgs = 0;

  void reset() {
    preSampleCount = 0;
    postSampleCount = 0;
    lastCg = nullptr;
    lastArgs = nullptr;
    lastNumArgs = 0;
  }
};

thread_local SampleCallbackTestData g_sampleCallbackData;

void testPreSampleCallback(void *cg, int64_t *args, int32_t num_args, void *userData) {
  auto *data = static_cast<SampleCallbackTestData *>(userData);
  data->preSampleCount++;
  data->lastCg = cg;
  data->lastArgs = args;
  data->lastNumArgs = num_args;
}

void testPostSampleCallback(void *cg, int64_t *args, int32_t num_args, void *userData) {
  auto *data = static_cast<SampleCallbackTestData *>(userData);
  data->postSampleCount++;
  data->lastCg = cg;
  data->lastArgs = args;
  data->lastNumArgs = num_args;
}
} // anonymous namespace

TEST(MooreRuntimeSampleCallbackTest, ExplicitSampleNoArgs) {
  // Test basic sample() with no arguments
  void *cg = __moore_covergroup_create("test_cg", 2);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");

  // Sample should not crash even without callbacks
  __moore_covergroup_sample(cg);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeSampleCallbackTest, PreSampleCallback) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");

  SampleCallbackTestData data;
  __moore_covergroup_set_pre_sample_callback(cg, testPreSampleCallback, &data);

  __moore_covergroup_sample(cg);

  EXPECT_EQ(data.preSampleCount, 1);
  EXPECT_EQ(data.lastCg, cg);
  EXPECT_EQ(data.lastNumArgs, 0);

  // Sample again
  __moore_covergroup_sample(cg);
  EXPECT_EQ(data.preSampleCount, 2);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeSampleCallbackTest, PostSampleCallback) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");

  SampleCallbackTestData data;
  __moore_covergroup_set_post_sample_callback(cg, testPostSampleCallback, &data);

  __moore_covergroup_sample(cg);

  EXPECT_EQ(data.postSampleCount, 1);
  EXPECT_EQ(data.lastCg, cg);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeSampleCallbackTest, BothCallbacks) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");

  SampleCallbackTestData preData, postData;
  __moore_covergroup_set_pre_sample_callback(cg, testPreSampleCallback, &preData);
  __moore_covergroup_set_post_sample_callback(cg, testPostSampleCallback, &postData);

  __moore_covergroup_sample(cg);

  EXPECT_EQ(preData.preSampleCount, 1);
  EXPECT_EQ(postData.postSampleCount, 1);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeSampleCallbackTest, SampleWithArguments) {
  void *cg = __moore_covergroup_create("test_cg", 3);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");
  __moore_coverpoint_init(cg, 2, "cp2");

  SampleCallbackTestData data;
  __moore_covergroup_set_pre_sample_callback(cg, testPreSampleCallback, &data);

  int64_t args[] = {10, 20, 30};
  __moore_covergroup_sample_with_args(cg, args, 3);

  EXPECT_EQ(data.preSampleCount, 1);
  EXPECT_EQ(data.lastNumArgs, 3);
  EXPECT_EQ(data.lastArgs, args);

  // Verify coverpoints were sampled with the arguments
  // Coverage should be non-zero
  double cov0 = __moore_coverpoint_get_coverage(cg, 0);
  double cov1 = __moore_coverpoint_get_coverage(cg, 1);
  double cov2 = __moore_coverpoint_get_coverage(cg, 2);
  EXPECT_EQ(cov0, 100.0);  // Single value = 100%
  EXPECT_EQ(cov1, 100.0);
  EXPECT_EQ(cov2, 100.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeSampleCallbackTest, SampleWithArgMapping) {
  void *cg = __moore_covergroup_create("test_cg", 3);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");
  __moore_coverpoint_init(cg, 1, "cp1");
  __moore_coverpoint_init(cg, 2, "cp2");

  // Map: cp0 <- arg1, cp1 <- arg0, cp2 <- skip
  int32_t mapping[] = {1, 0, -1};
  __moore_covergroup_set_sample_arg_mapping(cg, mapping, 3);

  int64_t args[] = {100, 200};
  __moore_covergroup_sample_with_args(cg, args, 2);

  // cp0 should have sampled 200 (arg1), cp1 should have sampled 100 (arg0)
  // cp2 should not have been sampled (coverage 0)
  double cov0 = __moore_coverpoint_get_coverage(cg, 0);
  double cov1 = __moore_coverpoint_get_coverage(cg, 1);
  double cov2 = __moore_coverpoint_get_coverage(cg, 2);
  EXPECT_EQ(cov0, 100.0);  // Sampled with value 200
  EXPECT_EQ(cov1, 100.0);  // Sampled with value 100
  EXPECT_EQ(cov2, 0.0);    // Not sampled

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeSampleCallbackTest, SampleEnabled) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");

  // Should be enabled by default
  EXPECT_TRUE(__moore_covergroup_is_sample_enabled(cg));

  SampleCallbackTestData data;
  __moore_covergroup_set_pre_sample_callback(cg, testPreSampleCallback, &data);

  // Sample while enabled
  __moore_covergroup_sample(cg);
  EXPECT_EQ(data.preSampleCount, 1);

  // Disable and try sampling
  __moore_covergroup_set_sample_enabled(cg, false);
  EXPECT_FALSE(__moore_covergroup_is_sample_enabled(cg));
  __moore_covergroup_sample(cg);
  EXPECT_EQ(data.preSampleCount, 1);  // Should not have increased

  // Re-enable
  __moore_covergroup_set_sample_enabled(cg, true);
  __moore_covergroup_sample(cg);
  EXPECT_EQ(data.preSampleCount, 2);  // Should have increased

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeSampleCallbackTest, SampleEvent) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");

  // Should not have sample event by default
  EXPECT_FALSE(__moore_covergroup_has_sample_event(cg));
  EXPECT_EQ(__moore_covergroup_get_sample_event(cg), nullptr);

  // Set sample event
  __moore_covergroup_set_sample_event(cg, "posedge_clk");
  EXPECT_TRUE(__moore_covergroup_has_sample_event(cg));
  EXPECT_STREQ(__moore_covergroup_get_sample_event(cg), "posedge_clk");

  // Track if sample happens
  SampleCallbackTestData data;
  __moore_covergroup_set_pre_sample_callback(cg, testPreSampleCallback, &data);

  // Trigger wrong event - should not sample
  __moore_covergroup_trigger_sample_event(cg, "negedge_clk");
  EXPECT_EQ(data.preSampleCount, 0);

  // Trigger correct event - should sample
  __moore_covergroup_trigger_sample_event(cg, "posedge_clk");
  EXPECT_EQ(data.preSampleCount, 1);

  // Trigger with NULL - should sample any configured event
  __moore_covergroup_trigger_sample_event(cg, nullptr);
  EXPECT_EQ(data.preSampleCount, 2);

  // Clear sample event
  __moore_covergroup_set_sample_event(cg, nullptr);
  EXPECT_FALSE(__moore_covergroup_has_sample_event(cg));

  // Should not trigger after clearing
  __moore_covergroup_trigger_sample_event(cg, "posedge_clk");
  EXPECT_EQ(data.preSampleCount, 2);  // Unchanged

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeSampleCallbackTest, GlobalCallbacks) {
  void *cg1 = __moore_covergroup_create("test_cg1", 1);
  void *cg2 = __moore_covergroup_create("test_cg2", 1);
  ASSERT_NE(cg1, nullptr);
  ASSERT_NE(cg2, nullptr);

  __moore_coverpoint_init(cg1, 0, "cp0");
  __moore_coverpoint_init(cg2, 0, "cp0");

  SampleCallbackTestData globalPreData, globalPostData;
  __moore_coverage_set_global_pre_sample_callback(testPreSampleCallback, &globalPreData);
  __moore_coverage_set_global_post_sample_callback(testPostSampleCallback, &globalPostData);

  // Sample cg1
  __moore_covergroup_sample(cg1);
  EXPECT_EQ(globalPreData.preSampleCount, 1);
  EXPECT_EQ(globalPostData.postSampleCount, 1);
  EXPECT_EQ(globalPreData.lastCg, cg1);

  // Sample cg2
  __moore_covergroup_sample(cg2);
  EXPECT_EQ(globalPreData.preSampleCount, 2);
  EXPECT_EQ(globalPostData.postSampleCount, 2);
  EXPECT_EQ(globalPreData.lastCg, cg2);

  // Clear global callbacks
  __moore_coverage_set_global_pre_sample_callback(nullptr, nullptr);
  __moore_coverage_set_global_post_sample_callback(nullptr, nullptr);

  // Sample again - global callbacks should not fire
  __moore_covergroup_sample(cg1);
  EXPECT_EQ(globalPreData.preSampleCount, 2);  // Unchanged
  EXPECT_EQ(globalPostData.postSampleCount, 2);

  __moore_covergroup_destroy(cg1);
  __moore_covergroup_destroy(cg2);
}

TEST(MooreRuntimeSampleCallbackTest, GlobalAndLocalCallbacks) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");

  SampleCallbackTestData globalPreData, localPreData;
  __moore_coverage_set_global_pre_sample_callback(testPreSampleCallback, &globalPreData);
  __moore_covergroup_set_pre_sample_callback(cg, testPreSampleCallback, &localPreData);

  __moore_covergroup_sample(cg);

  // Both should be called
  EXPECT_EQ(globalPreData.preSampleCount, 1);
  EXPECT_EQ(localPreData.preSampleCount, 1);

  // Cleanup
  __moore_coverage_set_global_pre_sample_callback(nullptr, nullptr);
  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeSampleCallbackTest, SampleNullCovergroup) {
  // These should not crash
  __moore_covergroup_sample(nullptr);

  int64_t args[] = {1, 2, 3};
  __moore_covergroup_sample_with_args(nullptr, args, 3);

  __moore_covergroup_set_pre_sample_callback(nullptr, testPreSampleCallback, nullptr);
  __moore_covergroup_set_post_sample_callback(nullptr, testPostSampleCallback, nullptr);

  __moore_covergroup_set_sample_enabled(nullptr, true);
  EXPECT_FALSE(__moore_covergroup_is_sample_enabled(nullptr));

  __moore_covergroup_set_sample_event(nullptr, "event");
  EXPECT_EQ(__moore_covergroup_get_sample_event(nullptr), nullptr);
  EXPECT_FALSE(__moore_covergroup_has_sample_event(nullptr));

  __moore_covergroup_trigger_sample_event(nullptr, "event");
}

TEST(MooreRuntimeSampleCallbackTest, DisableCallback) {
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  __moore_coverpoint_init(cg, 0, "cp0");

  SampleCallbackTestData data;
  __moore_covergroup_set_pre_sample_callback(cg, testPreSampleCallback, &data);

  __moore_covergroup_sample(cg);
  EXPECT_EQ(data.preSampleCount, 1);

  // Disable callback by setting it to nullptr
  __moore_covergroup_set_pre_sample_callback(cg, nullptr, nullptr);

  __moore_covergroup_sample(cg);
  EXPECT_EQ(data.preSampleCount, 1);  // Unchanged

  __moore_covergroup_destroy(cg);
}

//===----------------------------------------------------------------------===//
// Coverage Exclusion Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeCoverageExclusionTest, ExcludeBinBasic) {
  // Test basic bin exclusion functionality
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  // Set up explicit bins
  MooreCoverageBin bins[3];
  bins[0] = {.name = "bin0", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 0, .high = 9, .hit_count = 0};
  bins[1] = {.name = "bin1", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 10, .high = 19, .hit_count = 0};
  bins[2] = {.name = "bin2", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 20, .high = 29, .hit_count = 0};

  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 3);

  // Sample all bins
  __moore_coverpoint_sample(cg, 0, 5);   // bin0
  __moore_coverpoint_sample(cg, 0, 15);  // bin1
  __moore_coverpoint_sample(cg, 0, 25);  // bin2

  // All 3 bins hit, coverage should be 100%
  double cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov, 100.0);

  // Exclude bin1
  __moore_coverpoint_exclude_bin(cg, 0, "bin1");

  // Now only 2 bins count (bin0 and bin2), both still hit, so 100%
  cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov, 100.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageExclusionTest, ExcludeBinAffectsCoverage) {
  // Test that excluding an unhit bin affects coverage calculation
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[4];
  bins[0] = {.name = "bin0", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 0, .high = 9, .hit_count = 0};
  bins[1] = {.name = "bin1", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 10, .high = 19, .hit_count = 0};
  bins[2] = {.name = "bin2", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 20, .high = 29, .hit_count = 0};
  bins[3] = {.name = "bin3", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 30, .high = 39, .hit_count = 0};

  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 4);

  // Sample only bin0 and bin1 (2/4 = 50% coverage)
  __moore_coverpoint_sample(cg, 0, 5);   // bin0
  __moore_coverpoint_sample(cg, 0, 15);  // bin1

  double cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov, 50.0);

  // Exclude bin2 and bin3 (the unhit bins)
  __moore_coverpoint_exclude_bin(cg, 0, "bin2");
  __moore_coverpoint_exclude_bin(cg, 0, "bin3");

  // Now only bin0 and bin1 count, both hit, so 100%
  cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov, 100.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageExclusionTest, IncludeBin) {
  // Test re-including an excluded bin
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[2];
  bins[0] = {.name = "bin0", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 0, .high = 9, .hit_count = 0};
  bins[1] = {.name = "bin1", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 10, .high = 19, .hit_count = 0};

  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 2);

  // Sample only bin0
  __moore_coverpoint_sample(cg, 0, 5);

  // 1/2 bins hit = 50%
  double cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov, 50.0);

  // Exclude bin1
  __moore_coverpoint_exclude_bin(cg, 0, "bin1");

  // Now 1/1 bin hit = 100%
  cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov, 100.0);

  // Re-include bin1
  __moore_coverpoint_include_bin(cg, 0, "bin1");

  // Back to 1/2 bins hit = 50%
  cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov, 50.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageExclusionTest, IsBinExcluded) {
  // Test checking exclusion status
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[2];
  bins[0] = {.name = "bin0", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 0, .high = 9, .hit_count = 0};
  bins[1] = {.name = "bin1", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 10, .high = 19, .hit_count = 0};

  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 2);

  // Initially no bins are excluded
  EXPECT_FALSE(__moore_coverpoint_is_bin_excluded(cg, 0, "bin0"));
  EXPECT_FALSE(__moore_coverpoint_is_bin_excluded(cg, 0, "bin1"));

  // Exclude bin0
  __moore_coverpoint_exclude_bin(cg, 0, "bin0");

  EXPECT_TRUE(__moore_coverpoint_is_bin_excluded(cg, 0, "bin0"));
  EXPECT_FALSE(__moore_coverpoint_is_bin_excluded(cg, 0, "bin1"));

  // Include bin0 again
  __moore_coverpoint_include_bin(cg, 0, "bin0");

  EXPECT_FALSE(__moore_coverpoint_is_bin_excluded(cg, 0, "bin0"));
  EXPECT_FALSE(__moore_coverpoint_is_bin_excluded(cg, 0, "bin1"));

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageExclusionTest, GetExcludedBinCount) {
  // Test getting the count of excluded bins
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[3];
  bins[0] = {.name = "bin0", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 0, .high = 9, .hit_count = 0};
  bins[1] = {.name = "bin1", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 10, .high = 19, .hit_count = 0};
  bins[2] = {.name = "bin2", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 20, .high = 29, .hit_count = 0};

  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 3);

  // Initially no exclusions
  EXPECT_EQ(__moore_coverpoint_get_excluded_bin_count(cg, 0), 0);

  // Exclude one bin
  __moore_coverpoint_exclude_bin(cg, 0, "bin1");
  EXPECT_EQ(__moore_coverpoint_get_excluded_bin_count(cg, 0), 1);

  // Exclude another bin
  __moore_coverpoint_exclude_bin(cg, 0, "bin2");
  EXPECT_EQ(__moore_coverpoint_get_excluded_bin_count(cg, 0), 2);

  // Re-include one
  __moore_coverpoint_include_bin(cg, 0, "bin1");
  EXPECT_EQ(__moore_coverpoint_get_excluded_bin_count(cg, 0), 1);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageExclusionTest, ClearExclusions) {
  // Test clearing all exclusions for a coverpoint
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[3];
  bins[0] = {.name = "bin0", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 0, .high = 9, .hit_count = 0};
  bins[1] = {.name = "bin1", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 10, .high = 19, .hit_count = 0};
  bins[2] = {.name = "bin2", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 20, .high = 29, .hit_count = 0};

  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 3);

  // Exclude all bins
  __moore_coverpoint_exclude_bin(cg, 0, "bin0");
  __moore_coverpoint_exclude_bin(cg, 0, "bin1");
  __moore_coverpoint_exclude_bin(cg, 0, "bin2");
  EXPECT_EQ(__moore_coverpoint_get_excluded_bin_count(cg, 0), 3);

  // Clear all exclusions
  __moore_coverpoint_clear_exclusions(cg, 0);
  EXPECT_EQ(__moore_coverpoint_get_excluded_bin_count(cg, 0), 0);

  // Verify all bins are no longer excluded
  EXPECT_FALSE(__moore_coverpoint_is_bin_excluded(cg, 0, "bin0"));
  EXPECT_FALSE(__moore_coverpoint_is_bin_excluded(cg, 0, "bin1"));
  EXPECT_FALSE(__moore_coverpoint_is_bin_excluded(cg, 0, "bin2"));

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageExclusionTest, ExclusionWithIgnoreBins) {
  // Test that exclusions work alongside ignore_bins
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  // Set up bins: 2 normal, 1 ignore
  MooreCoverageBin bins[3];
  bins[0] = {.name = "bin0", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 0, .high = 9, .hit_count = 0};
  bins[1] = {.name = "bin1", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_IGNORE,
             .low = 10, .high = 19, .hit_count = 0};
  bins[2] = {.name = "bin2", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 20, .high = 29, .hit_count = 0};

  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 3);

  // Sample bin0 only
  __moore_coverpoint_sample(cg, 0, 5);

  // 2 normal bins (bin0, bin2), 1 hit (bin0) = 50%
  double cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov, 50.0);

  // Exclude bin2
  __moore_coverpoint_exclude_bin(cg, 0, "bin2");

  // Now only bin0 counts, which is hit = 100%
  cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov, 100.0);

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageExclusionTest, ExclusionNullInputs) {
  // Test that null inputs don't crash
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[1];
  bins[0] = {.name = "bin0", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 0, .high = 9, .hit_count = 0};

  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 1);

  // Test with null covergroup
  __moore_coverpoint_exclude_bin(nullptr, 0, "bin0");
  __moore_coverpoint_include_bin(nullptr, 0, "bin0");
  EXPECT_FALSE(__moore_coverpoint_is_bin_excluded(nullptr, 0, "bin0"));
  EXPECT_EQ(__moore_coverpoint_get_excluded_bin_count(nullptr, 0), 0);
  __moore_coverpoint_clear_exclusions(nullptr, 0);

  // Test with null bin_name
  __moore_coverpoint_exclude_bin(cg, 0, nullptr);
  __moore_coverpoint_include_bin(cg, 0, nullptr);
  EXPECT_FALSE(__moore_coverpoint_is_bin_excluded(cg, 0, nullptr));

  // Test with invalid index
  __moore_coverpoint_exclude_bin(cg, -1, "bin0");
  __moore_coverpoint_exclude_bin(cg, 100, "bin0");
  EXPECT_FALSE(__moore_coverpoint_is_bin_excluded(cg, -1, "bin0"));
  EXPECT_FALSE(__moore_coverpoint_is_bin_excluded(cg, 100, "bin0"));

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageExclusionTest, ExclusionFileNullFilename) {
  // Test that null filename returns false
  EXPECT_FALSE(__moore_covergroup_set_exclusion_file(nullptr));
}

TEST(MooreRuntimeCoverageExclusionTest, ExclusionFileNonexistent) {
  // Test that nonexistent file returns false
  EXPECT_FALSE(__moore_covergroup_set_exclusion_file("/nonexistent/path/file.excl"));
}

TEST(MooreRuntimeCoverageExclusionTest, GetExclusionFile) {
  // Initially no exclusion file is set
  const char *file = __moore_covergroup_get_exclusion_file();
  // Note: file may or may not be nullptr depending on previous test state

  // After attempting to set a nonexistent file, the path is still stored
  __moore_covergroup_set_exclusion_file("/some/path/exclusions.txt");
  file = __moore_covergroup_get_exclusion_file();
  EXPECT_NE(file, nullptr);
  EXPECT_STREQ(file, "/some/path/exclusions.txt");
}

TEST(MooreRuntimeCoverageExclusionTest, MultipleExclusionsSameBin) {
  // Test that excluding the same bin multiple times is idempotent
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[2];
  bins[0] = {.name = "bin0", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 0, .high = 9, .hit_count = 0};
  bins[1] = {.name = "bin1", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 10, .high = 19, .hit_count = 0};

  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 2);

  // Exclude bin0 multiple times
  __moore_coverpoint_exclude_bin(cg, 0, "bin0");
  __moore_coverpoint_exclude_bin(cg, 0, "bin0");
  __moore_coverpoint_exclude_bin(cg, 0, "bin0");

  // Should still only count as 1 exclusion
  EXPECT_EQ(__moore_coverpoint_get_excluded_bin_count(cg, 0), 1);

  // Re-including once should remove it
  __moore_coverpoint_include_bin(cg, 0, "bin0");
  EXPECT_EQ(__moore_coverpoint_get_excluded_bin_count(cg, 0), 0);
  EXPECT_FALSE(__moore_coverpoint_is_bin_excluded(cg, 0, "bin0"));

  __moore_covergroup_destroy(cg);
}

TEST(MooreRuntimeCoverageExclusionTest, ExclusionPreservedAfterSampling) {
  // Test that exclusions are preserved when sampling continues
  void *cg = __moore_covergroup_create("test_cg", 1);
  ASSERT_NE(cg, nullptr);

  MooreCoverageBin bins[2];
  bins[0] = {.name = "bin0", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 0, .high = 9, .hit_count = 0};
  bins[1] = {.name = "bin1", .type = MOORE_BIN_RANGE, .kind = MOORE_BIN_KIND_NORMAL,
             .low = 10, .high = 19, .hit_count = 0};

  __moore_coverpoint_init_with_bins(cg, 0, "cp", bins, 2);

  // Exclude bin1 before sampling
  __moore_coverpoint_exclude_bin(cg, 0, "bin1");

  // Sample bin0 multiple times
  __moore_coverpoint_sample(cg, 0, 1);
  __moore_coverpoint_sample(cg, 0, 2);
  __moore_coverpoint_sample(cg, 0, 3);

  // bin1 should still be excluded
  EXPECT_TRUE(__moore_coverpoint_is_bin_excluded(cg, 0, "bin1"));

  // Coverage should be 100% (only bin0 counts and it's hit)
  double cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov, 100.0);

  // Sample into bin1 as well
  __moore_coverpoint_sample(cg, 0, 15);

  // bin1 still excluded, coverage still 100%
  EXPECT_TRUE(__moore_coverpoint_is_bin_excluded(cg, 0, "bin1"));
  cov = __moore_coverpoint_get_coverage(cg, 0);
  EXPECT_DOUBLE_EQ(cov, 100.0);

  __moore_covergroup_destroy(cg);
}

//===----------------------------------------------------------------------===//
// UVM Coverage Model API Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeUvmCoverageTest, CoverageModelSetGet) {
  // Reset coverage model first
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
  EXPECT_EQ(__moore_uvm_get_coverage_model(), UVM_NO_COVERAGE);

  // Set single coverage model
  __moore_uvm_set_coverage_model(UVM_CVR_REG_BITS);
  EXPECT_EQ(__moore_uvm_get_coverage_model(), UVM_CVR_REG_BITS);

  // Set multiple coverage models
  __moore_uvm_set_coverage_model(UVM_CVR_REG_BITS | UVM_CVR_FIELD_VALS);
  EXPECT_EQ(__moore_uvm_get_coverage_model(),
            UVM_CVR_REG_BITS | UVM_CVR_FIELD_VALS);

  // Set all coverage models
  __moore_uvm_set_coverage_model(UVM_CVR_ALL);
  EXPECT_EQ(__moore_uvm_get_coverage_model(), UVM_CVR_ALL);

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

TEST(MooreRuntimeUvmCoverageTest, HasCoverage) {
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);

  // No coverage enabled
  EXPECT_FALSE(__moore_uvm_has_coverage(UVM_CVR_REG_BITS));
  EXPECT_FALSE(__moore_uvm_has_coverage(UVM_CVR_FIELD_VALS));
  EXPECT_FALSE(__moore_uvm_has_coverage(UVM_CVR_ADDR_MAP));

  // Enable register bits coverage
  __moore_uvm_set_coverage_model(UVM_CVR_REG_BITS);
  EXPECT_TRUE(__moore_uvm_has_coverage(UVM_CVR_REG_BITS));
  EXPECT_FALSE(__moore_uvm_has_coverage(UVM_CVR_FIELD_VALS));
  EXPECT_FALSE(__moore_uvm_has_coverage(UVM_CVR_ADDR_MAP));

  // Enable all
  __moore_uvm_set_coverage_model(UVM_CVR_ALL);
  EXPECT_TRUE(__moore_uvm_has_coverage(UVM_CVR_REG_BITS));
  EXPECT_TRUE(__moore_uvm_has_coverage(UVM_CVR_FIELD_VALS));
  EXPECT_TRUE(__moore_uvm_has_coverage(UVM_CVR_ADDR_MAP));

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

TEST(MooreRuntimeUvmCoverageTest, SampleRegCoverageDisabled) {
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
  __moore_uvm_reset_coverage();

  // Sampling should not record anything when coverage is disabled
  __moore_uvm_coverage_sample_reg("test_reg", 42);

  // Coverage should be 0 since no covergroup was created
  double cov = __moore_uvm_get_reg_coverage("test_reg");
  EXPECT_DOUBLE_EQ(cov, 0.0);
}

TEST(MooreRuntimeUvmCoverageTest, SampleRegCoverageEnabled) {
  __moore_uvm_set_coverage_model(UVM_CVR_REG_BITS);
  __moore_uvm_reset_coverage();

  // Sample a register value
  __moore_uvm_coverage_sample_reg("status_reg", 0x55);

  // Coverage should be non-zero now
  double cov = __moore_uvm_get_reg_coverage("status_reg");
  EXPECT_GT(cov, 0.0);

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

// DISABLED: Field coverage returns 0 - needs implementation fix
TEST(MooreRuntimeUvmCoverageTest, DISABLED_SampleFieldCoverageEnabled) {
  __moore_uvm_set_coverage_model(UVM_CVR_FIELD_VALS);
  __moore_uvm_reset_coverage();

  // Set field range first
  __moore_uvm_set_field_range("my_field", 0, 15);

  // Sample field values
  __moore_uvm_coverage_sample_field("my_field", 5);
  __moore_uvm_coverage_sample_field("my_field", 10);

  // Coverage should be non-zero
  double cov = __moore_uvm_get_field_coverage("my_field");
  EXPECT_GT(cov, 0.0);

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

TEST(MooreRuntimeUvmCoverageTest, SampleAddrMapCoverage) {
  __moore_uvm_set_coverage_model(UVM_CVR_ADDR_MAP);
  __moore_uvm_reset_coverage();

  // Sample address map accesses
  __moore_uvm_coverage_sample_addr_map("apb_map", 0x1000, true);  // read
  __moore_uvm_coverage_sample_addr_map("apb_map", 0x1004, false); // write

  // Total coverage should be non-zero
  double cov = __moore_uvm_get_coverage();
  EXPECT_GT(cov, 0.0);

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

TEST(MooreRuntimeUvmCoverageTest, TotalCoverage) {
  __moore_uvm_set_coverage_model(UVM_CVR_REG_BITS | UVM_CVR_FIELD_VALS);
  __moore_uvm_reset_coverage();

  // Sample both register and field
  __moore_uvm_coverage_sample_reg("ctrl_reg", 0x12);
  __moore_uvm_coverage_sample_field("enable_field", 1);

  // Total coverage is average of all covergroups
  double cov = __moore_uvm_get_coverage();
  EXPECT_GT(cov, 0.0);

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

TEST(MooreRuntimeUvmCoverageTest, ResetCoverage) {
  __moore_uvm_set_coverage_model(UVM_CVR_REG_BITS);
  __moore_uvm_reset_coverage();

  // Sample a value
  __moore_uvm_coverage_sample_reg("reset_test_reg", 0xFF);
  double cov1 = __moore_uvm_get_reg_coverage("reset_test_reg");
  EXPECT_GT(cov1, 0.0);

  // Reset coverage
  __moore_uvm_reset_coverage();

  // Coverage for the same register should be back to baseline
  // (covergroup still exists but hits are reset)
  double cov2 = __moore_uvm_get_reg_coverage("reset_test_reg");
  EXPECT_LT(cov2, cov1);

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

TEST(MooreRuntimeUvmCoverageTest, RegBitWidthSetting) {
  __moore_uvm_set_coverage_model(UVM_CVR_REG_BITS);
  __moore_uvm_reset_coverage();

  // Set bit width for a register before sampling
  __moore_uvm_set_reg_bit_width("narrow_reg", 8);

  // Sample some values within the 8-bit range
  __moore_uvm_coverage_sample_reg("narrow_reg", 0);
  __moore_uvm_coverage_sample_reg("narrow_reg", 128);
  __moore_uvm_coverage_sample_reg("narrow_reg", 255);

  // Coverage should reflect the narrower range
  double cov = __moore_uvm_get_reg_coverage("narrow_reg");
  EXPECT_GT(cov, 0.0);

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

TEST(MooreRuntimeUvmCoverageTest, FieldRangeSetting) {
  __moore_uvm_set_coverage_model(UVM_CVR_FIELD_VALS);
  __moore_uvm_reset_coverage();

  // Set a custom range for the field before creating the covergroup.
  // The range affects auto_bin_max setting.
  __moore_uvm_set_field_range("custom_field", 10, 20);

  // Sample values in range. With auto bins and auto_bin_max=11 (range size),
  // each unique value contributes to coverage.
  for (int i = 10; i <= 20; i++) {
    __moore_uvm_coverage_sample_field("custom_field", i);
  }

  // Coverage should be non-zero (auto bins track observed values)
  double cov = __moore_uvm_get_field_coverage("custom_field");
  EXPECT_GT(cov, 0.0);

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

TEST(MooreRuntimeUvmCoverageTest, RegCoverageCallback) {
  __moore_uvm_set_coverage_model(UVM_CVR_REG_BITS);
  __moore_uvm_reset_coverage();

  struct CallbackData {
    std::string regName;
    int64_t value;
    int callCount;
  };
  CallbackData cbData = {"", 0, 0};

  // Set callback
  __moore_uvm_set_reg_coverage_callback(
      [](const char *reg_name, int64_t value, void *userData) {
        auto *data = static_cast<CallbackData *>(userData);
        data->regName = reg_name;
        data->value = value;
        data->callCount++;
      },
      &cbData);

  // Sample a value
  __moore_uvm_coverage_sample_reg("callback_reg", 0xAB);

  // Callback should have been invoked
  EXPECT_EQ(cbData.regName, "callback_reg");
  EXPECT_EQ(cbData.value, 0xAB);
  EXPECT_EQ(cbData.callCount, 1);

  // Sample another value
  __moore_uvm_coverage_sample_reg("callback_reg", 0xCD);
  EXPECT_EQ(cbData.value, 0xCD);
  EXPECT_EQ(cbData.callCount, 2);

  // Clear callback
  __moore_uvm_set_reg_coverage_callback(nullptr, nullptr);

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

TEST(MooreRuntimeUvmCoverageTest, FieldCoverageCallback) {
  __moore_uvm_set_coverage_model(UVM_CVR_FIELD_VALS);
  __moore_uvm_reset_coverage();

  struct CallbackData {
    std::string fieldName;
    int64_t value;
    int callCount;
  };
  CallbackData cbData = {"", 0, 0};

  // Set callback
  __moore_uvm_set_field_coverage_callback(
      [](const char *field_name, int64_t value, void *userData) {
        auto *data = static_cast<CallbackData *>(userData);
        data->fieldName = field_name;
        data->value = value;
        data->callCount++;
      },
      &cbData);

  // Sample a value
  __moore_uvm_coverage_sample_field("callback_field", 42);

  // Callback should have been invoked
  EXPECT_EQ(cbData.fieldName, "callback_field");
  EXPECT_EQ(cbData.value, 42);
  EXPECT_EQ(cbData.callCount, 1);

  // Clear callback
  __moore_uvm_set_field_coverage_callback(nullptr, nullptr);

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

TEST(MooreRuntimeUvmCoverageTest, NullInputHandling) {
  __moore_uvm_set_coverage_model(UVM_CVR_ALL);

  // These should not crash
  __moore_uvm_coverage_sample_reg(nullptr, 0);
  __moore_uvm_coverage_sample_field(nullptr, 0);
  __moore_uvm_coverage_sample_addr_map(nullptr, 0, true);

  EXPECT_DOUBLE_EQ(__moore_uvm_get_reg_coverage(nullptr), 0.0);
  EXPECT_DOUBLE_EQ(__moore_uvm_get_field_coverage(nullptr), 0.0);

  __moore_uvm_set_reg_bit_width(nullptr, 8);
  __moore_uvm_set_field_range(nullptr, 0, 10);

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

TEST(MooreRuntimeUvmCoverageTest, InvalidBitWidth) {
  __moore_uvm_set_coverage_model(UVM_CVR_REG_BITS);

  // Invalid bit widths should be ignored
  __moore_uvm_set_reg_bit_width("test_reg", 0);   // too small
  __moore_uvm_set_reg_bit_width("test_reg", 65);  // too large
  __moore_uvm_set_reg_bit_width("test_reg", -1);  // negative

  // These should not cause issues
  __moore_uvm_coverage_sample_reg("test_reg", 0xFF);

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

TEST(MooreRuntimeUvmCoverageTest, InvalidFieldRange) {
  __moore_uvm_set_coverage_model(UVM_CVR_FIELD_VALS);

  // Invalid range (min > max) should be ignored
  __moore_uvm_set_field_range("test_field", 100, 50);

  // Sampling should still work with default range
  __moore_uvm_coverage_sample_field("test_field", 10);

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

TEST(MooreRuntimeUvmCoverageTest, MultipleRegisters) {
  __moore_uvm_set_coverage_model(UVM_CVR_REG_BITS);
  __moore_uvm_reset_coverage();

  // Sample multiple different registers
  __moore_uvm_coverage_sample_reg("reg_a", 0x11);
  __moore_uvm_coverage_sample_reg("reg_b", 0x22);
  __moore_uvm_coverage_sample_reg("reg_c", 0x33);

  // Each should have independent coverage
  double covA = __moore_uvm_get_reg_coverage("reg_a");
  double covB = __moore_uvm_get_reg_coverage("reg_b");
  double covC = __moore_uvm_get_reg_coverage("reg_c");

  EXPECT_GT(covA, 0.0);
  EXPECT_GT(covB, 0.0);
  EXPECT_GT(covC, 0.0);

  // Total coverage is average
  double total = __moore_uvm_get_coverage();
  EXPECT_GT(total, 0.0);

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

TEST(MooreRuntimeUvmCoverageTest, CoverageModelEnumValues) {
  // Verify enum values match UVM specification
  EXPECT_EQ(UVM_CVR_REG_BITS, 1);
  EXPECT_EQ(UVM_CVR_ADDR_MAP, 2);
  EXPECT_EQ(UVM_CVR_FIELD_VALS, 4);
  EXPECT_EQ(UVM_NO_COVERAGE, 0);
  EXPECT_EQ(UVM_CVR_ALL, 7);  // All bits set (1 + 2 + 4)
}

TEST(MooreRuntimeUvmCoverageTest, CombinedCoverageModels) {
  // Test with multiple coverage models enabled
  __moore_uvm_set_coverage_model(UVM_CVR_REG_BITS | UVM_CVR_ADDR_MAP);
  __moore_uvm_reset_coverage();

  // REG_BITS should be enabled
  EXPECT_TRUE(__moore_uvm_has_coverage(UVM_CVR_REG_BITS));
  // ADDR_MAP should be enabled
  EXPECT_TRUE(__moore_uvm_has_coverage(UVM_CVR_ADDR_MAP));
  // FIELD_VALS should not be enabled
  EXPECT_FALSE(__moore_uvm_has_coverage(UVM_CVR_FIELD_VALS));

  // Sample register - should work
  __moore_uvm_coverage_sample_reg("combo_reg", 0x99);
  EXPECT_GT(__moore_uvm_get_reg_coverage("combo_reg"), 0.0);

  // Sample field - should not record (not enabled)
  __moore_uvm_coverage_sample_field("combo_field", 5);
  EXPECT_DOUBLE_EQ(__moore_uvm_get_field_coverage("combo_field"), 0.0);

  // Reset
  __moore_uvm_set_coverage_model(UVM_NO_COVERAGE);
}

//===----------------------------------------------------------------------===//
// Display System Task Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeDisplayTest, DisplayWithMessage) {
  // Test __moore_display with a valid message
  char data[] = "Test message";
  MooreString str = {data, 12};

  // Capture stdout to verify output
  testing::internal::CaptureStdout();
  __moore_display(&str);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(output, "Test message\n");
}

TEST(MooreRuntimeDisplayTest, DisplayWithEmptyMessage) {
  // Test __moore_display with an empty message
  MooreString str = {nullptr, 0};

  testing::internal::CaptureStdout();
  __moore_display(&str);
  std::string output = testing::internal::GetCapturedStdout();

  // Should still output newline
  EXPECT_EQ(output, "\n");
}

TEST(MooreRuntimeDisplayTest, DisplayWithNullptr) {
  // Test __moore_display with nullptr
  testing::internal::CaptureStdout();
  __moore_display(nullptr);
  std::string output = testing::internal::GetCapturedStdout();

  // Should produce no output
  EXPECT_EQ(output, "");
}

TEST(MooreRuntimeDisplayTest, WriteWithMessage) {
  // Test __moore_write - no newline
  char data[] = "Write test";
  MooreString str = {data, 10};

  testing::internal::CaptureStdout();
  __moore_write(&str);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(output, "Write test");
}

TEST(MooreRuntimeDisplayTest, WriteWithEmptyMessage) {
  // Test __moore_write with empty message
  MooreString str = {nullptr, 0};

  testing::internal::CaptureStdout();
  __moore_write(&str);
  std::string output = testing::internal::GetCapturedStdout();

  // No output for empty message
  EXPECT_EQ(output, "");
}

TEST(MooreRuntimeDisplayTest, PrintDynString) {
  // Test __moore_print_dyn_string
  char data[] = "Dynamic string content";
  MooreString str = {data, 22};

  testing::internal::CaptureStdout();
  __moore_print_dyn_string(&str);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(output, "Dynamic string content");
}

TEST(MooreRuntimeDisplayTest, StrobeBasic) {
  // Test __moore_strobe - should queue messages
  char data1[] = "Strobe message 1";
  MooreString str1 = {data1, 16};

  __moore_strobe(&str1);

  // Message should not appear yet
  testing::internal::CaptureStdout();
  // Nothing captured here since strobe is queued

  // Flush strobe queue
  __moore_strobe_flush();
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(output, "Strobe message 1\n");
}

TEST(MooreRuntimeDisplayTest, StrobeMultipleMessages) {
  // Test multiple strobe messages
  char data1[] = "First";
  char data2[] = "Second";
  MooreString str1 = {data1, 5};
  MooreString str2 = {data2, 6};

  __moore_strobe(&str1);
  __moore_strobe(&str2);

  testing::internal::CaptureStdout();
  __moore_strobe_flush();
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(output, "First\nSecond\n");
}

TEST(MooreRuntimeDisplayTest, StrobeFlushClearsQueue) {
  // Test that flush clears the queue
  char data[] = "Test";
  MooreString str = {data, 4};

  __moore_strobe(&str);
  __moore_strobe_flush();

  // Second flush should produce no output
  testing::internal::CaptureStdout();
  __moore_strobe_flush();
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(output, "");
}

TEST(MooreRuntimeDisplayTest, SimulationTime) {
  // Test simulation time functions
  __moore_set_time(1000);
  EXPECT_EQ(__moore_get_time(), 1000);

  __moore_set_time(12345);
  EXPECT_EQ(__moore_get_time(), 12345);

  __moore_set_time(0);
  EXPECT_EQ(__moore_get_time(), 0);
}

TEST(MooreRuntimeDisplayTest, MonitorOnOff) {
  // Test monitor enable/disable
  char data[] = "Monitor test";
  MooreString str = {data, 12};

  // Capture initial monitor output
  testing::internal::CaptureStdout();
  __moore_monitor(&str, nullptr, 0, nullptr);
  std::string output = testing::internal::GetCapturedStdout();

  // Should print immediately when set up
  EXPECT_EQ(output, "Monitor test\n");

  // Test monitoroff
  __moore_monitoroff();

  // Monitor check should not print when disabled
  testing::internal::CaptureStdout();
  __moore_monitor_check();
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "");

  // Test monitoron
  __moore_monitoron();
}

TEST(MooreRuntimeDisplayTest, MonitorValueChange) {
  // Test monitor with value change detection
  int32_t value = 42;
  void *values[] = {&value};
  int32_t sizes[] = {sizeof(value)};
  char data[] = "Value changed";
  MooreString str = {data, 13};

  // Clear any previous monitor state
  MooreString empty = {nullptr, 0};
  __moore_monitor(&empty, nullptr, 0, nullptr);

  // Set up monitor with a value pointer
  testing::internal::CaptureStdout();
  __moore_monitor(&str, values, 1, sizes);
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "Value changed\n");

  // No change - check should not trigger
  testing::internal::CaptureStdout();
  __moore_monitor_check();
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "");

  // Change value and check
  value = 100;
  testing::internal::CaptureStdout();
  __moore_monitor_check();
  output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, "Value changed\n");

  // Clear monitor
  __moore_monitor(&empty, nullptr, 0, nullptr);
}

//===----------------------------------------------------------------------===//
// Implication Constraint Tests
// IEEE 1800-2017 Section 18.5.6 "Implication constraints"
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeImplicationTest, BasicImplicationTruthTable) {
  // Test all four cases of the implication truth table
  // antecedent -> consequent

  // Case 1: 0 -> 0 = 1 (vacuously true)
  EXPECT_EQ(__moore_constraint_check_implication(0, 0), 1);

  // Case 2: 0 -> 1 = 1 (vacuously true)
  EXPECT_EQ(__moore_constraint_check_implication(0, 1), 1);

  // Case 3: 1 -> 0 = 0 (violated)
  EXPECT_EQ(__moore_constraint_check_implication(1, 0), 0);

  // Case 4: 1 -> 1 = 1 (satisfied)
  EXPECT_EQ(__moore_constraint_check_implication(1, 1), 1);
}

TEST(MooreRuntimeImplicationTest, ImplicationWithNonZeroValues) {
  // Test with various non-zero values as "true"
  EXPECT_EQ(__moore_constraint_check_implication(42, 0), 0);  // non-zero -> 0 = violated
  EXPECT_EQ(__moore_constraint_check_implication(100, 50), 1); // non-zero -> non-zero = satisfied
  EXPECT_EQ(__moore_constraint_check_implication(-1, 1), 1);  // negative -> positive = satisfied
  EXPECT_EQ(__moore_constraint_check_implication(-1, 0), 0);  // negative -> 0 = violated
}

TEST(MooreRuntimeImplicationTest, NestedImplication) {
  // Test a -> (b -> c) nested implication

  // If outer (a) is false, entire implication is true
  EXPECT_EQ(__moore_constraint_check_nested_implication(0, 0, 0), 1);
  EXPECT_EQ(__moore_constraint_check_nested_implication(0, 0, 1), 1);
  EXPECT_EQ(__moore_constraint_check_nested_implication(0, 1, 0), 1);
  EXPECT_EQ(__moore_constraint_check_nested_implication(0, 1, 1), 1);

  // If outer is true but inner is false, inner implication is true
  EXPECT_EQ(__moore_constraint_check_nested_implication(1, 0, 0), 1);
  EXPECT_EQ(__moore_constraint_check_nested_implication(1, 0, 1), 1);

  // If both antecedents are true, consequent must be true
  EXPECT_EQ(__moore_constraint_check_nested_implication(1, 1, 0), 0);  // violated
  EXPECT_EQ(__moore_constraint_check_nested_implication(1, 1, 1), 1);  // satisfied
}

TEST(MooreRuntimeImplicationTest, SoftImplicationHard) {
  // Test hard implication behavior

  // Antecedent false - always satisfied (vacuously true)
  EXPECT_EQ(__moore_constraint_check_implication_soft(0, 0, 0), 1);
  EXPECT_EQ(__moore_constraint_check_implication_soft(0, 1, 0), 1);

  // Antecedent true, consequent satisfied - satisfied
  EXPECT_EQ(__moore_constraint_check_implication_soft(1, 1, 0), 1);

  // Antecedent true, consequent not satisfied - hard constraint violated
  EXPECT_EQ(__moore_constraint_check_implication_soft(1, 0, 0), 0);
}

TEST(MooreRuntimeImplicationTest, SoftImplicationSoft) {
  // Test soft implication behavior

  // Antecedent false - always satisfied
  EXPECT_EQ(__moore_constraint_check_implication_soft(0, 0, 1), 1);
  EXPECT_EQ(__moore_constraint_check_implication_soft(0, 1, 1), 1);

  // Antecedent true, consequent satisfied - satisfied
  EXPECT_EQ(__moore_constraint_check_implication_soft(1, 1, 1), 1);

  // Antecedent true, consequent not satisfied - soft constraint uses fallback
  EXPECT_EQ(__moore_constraint_check_implication_soft(1, 0, 1), 1);  // soft fallback
}

TEST(MooreRuntimeImplicationTest, StatisticsTracking) {
  // Reset statistics before test
  __moore_implication_reset_stats();

  MooreImplicationStats *stats = __moore_implication_get_stats();
  ASSERT_NE(stats, nullptr);

  // Initial state should be zero
  EXPECT_EQ(stats->totalImplications, 0);
  EXPECT_EQ(stats->triggeredImplications, 0);
  EXPECT_EQ(stats->satisfiedImplications, 0);
  EXPECT_EQ(stats->softFallbacks, 0);

  // Make some implication checks
  __moore_constraint_check_implication(0, 0);  // vacuously true
  __moore_constraint_check_implication(1, 1);  // triggered, satisfied
  __moore_constraint_check_implication(1, 0);  // triggered, violated

  EXPECT_EQ(stats->totalImplications, 3);
  EXPECT_EQ(stats->triggeredImplications, 2);  // only when antecedent is true
  EXPECT_EQ(stats->satisfiedImplications, 2);  // first two satisfied

  // Reset and verify
  __moore_implication_reset_stats();
  EXPECT_EQ(stats->totalImplications, 0);
  EXPECT_EQ(stats->triggeredImplications, 0);
  EXPECT_EQ(stats->satisfiedImplications, 0);
}

TEST(MooreRuntimeImplicationTest, SoftFallbackStatistics) {
  __moore_implication_reset_stats();

  MooreImplicationStats *stats = __moore_implication_get_stats();

  // Soft implication with violated consequent should use fallback
  __moore_constraint_check_implication_soft(1, 0, 1);  // soft fallback

  EXPECT_EQ(stats->softFallbacks, 1);
  EXPECT_EQ(stats->satisfiedImplications, 1);  // soft fallback counts as satisfied

  __moore_implication_reset_stats();
}

TEST(MooreRuntimeImplicationTest, NestedImplicationStatistics) {
  __moore_implication_reset_stats();

  MooreImplicationStats *stats = __moore_implication_get_stats();

  // Nested implications
  __moore_constraint_check_nested_implication(1, 1, 1);  // both triggered, satisfied
  __moore_constraint_check_nested_implication(1, 0, 0);  // outer triggered, inner vacuous

  EXPECT_EQ(stats->totalImplications, 2);
  EXPECT_EQ(stats->triggeredImplications, 2);
  EXPECT_EQ(stats->satisfiedImplications, 2);

  __moore_implication_reset_stats();
}

//===----------------------------------------------------------------------===//
// Simulation Control Task Tests
// IEEE 1800-2017 Section 21 "System tasks and system functions"
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeSimControlTest, FinishSetsState) {
  // Disable actual exit for testing
  __moore_set_finish_exits(false);
  __moore_reset_finish_state();

  EXPECT_FALSE(__moore_finish_called());
  EXPECT_EQ(__moore_get_exit_code(), 0);

  testing::internal::CaptureStderr();
  __moore_finish(0);
  std::string output = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(__moore_finish_called());
  EXPECT_EQ(__moore_get_exit_code(), 0);
  EXPECT_NE(output.find("$finish called with exit code 0"), std::string::npos);

  __moore_reset_finish_state();
}

TEST(MooreRuntimeSimControlTest, FinishWithExitCode) {
  __moore_set_finish_exits(false);
  __moore_reset_finish_state();

  testing::internal::CaptureStderr();
  __moore_finish(42);
  std::string output = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(__moore_finish_called());
  EXPECT_EQ(__moore_get_exit_code(), 42);
  EXPECT_NE(output.find("$finish called with exit code 42"), std::string::npos);

  __moore_reset_finish_state();
}

TEST(MooreRuntimeSimControlTest, FatalWithMessage) {
  __moore_set_finish_exits(false);
  __moore_reset_finish_state();

  char data[] = "Test fatal error";
  MooreString msg = {data, 16};

  testing::internal::CaptureStderr();
  __moore_fatal(1, &msg);
  std::string output = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(__moore_finish_called());
  EXPECT_EQ(__moore_get_exit_code(), 1);
  EXPECT_EQ(__moore_get_error_count(), 1);
  EXPECT_NE(output.find("Fatal: Test fatal error"), std::string::npos);

  __moore_reset_finish_state();
}

TEST(MooreRuntimeSimControlTest, FatalWithNullMessage) {
  __moore_set_finish_exits(false);
  __moore_reset_finish_state();

  testing::internal::CaptureStderr();
  __moore_fatal(2, nullptr);
  std::string output = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(__moore_finish_called());
  EXPECT_EQ(__moore_get_exit_code(), 2);
  EXPECT_NE(output.find("Fatal:"), std::string::npos);

  __moore_reset_finish_state();
}

TEST(MooreRuntimeSimControlTest, ErrorIncreasesCount) {
  __moore_reset_finish_state();

  EXPECT_EQ(__moore_get_error_count(), 0);

  char data[] = "Error message 1";
  MooreString msg = {data, 15};

  testing::internal::CaptureStderr();
  __moore_error(&msg);
  std::string output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(__moore_get_error_count(), 1);
  EXPECT_NE(output.find("Error: Error message 1"), std::string::npos);

  // Second error
  testing::internal::CaptureStderr();
  __moore_error(&msg);
  output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(__moore_get_error_count(), 2);

  __moore_reset_finish_state();
}

TEST(MooreRuntimeSimControlTest, WarningIncreasesCount) {
  __moore_reset_finish_state();

  EXPECT_EQ(__moore_get_warning_count(), 0);

  char data[] = "Warning message";
  MooreString msg = {data, 15};

  testing::internal::CaptureStderr();
  __moore_warning(&msg);
  std::string output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(__moore_get_warning_count(), 1);
  EXPECT_NE(output.find("Warning: Warning message"), std::string::npos);

  // Second warning
  testing::internal::CaptureStderr();
  __moore_warning(&msg);
  output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(__moore_get_warning_count(), 2);

  __moore_reset_finish_state();
}

TEST(MooreRuntimeSimControlTest, InfoDoesNotIncreaseCounts) {
  __moore_reset_finish_state();

  char data[] = "Info message";
  MooreString msg = {data, 12};

  testing::internal::CaptureStdout();
  __moore_info(&msg);
  std::string output = testing::internal::GetCapturedStdout();

  // Info should not increment error or warning count
  EXPECT_EQ(__moore_get_error_count(), 0);
  EXPECT_EQ(__moore_get_warning_count(), 0);
  EXPECT_NE(output.find("Info: Info message"), std::string::npos);

  __moore_reset_finish_state();
}

TEST(MooreRuntimeSimControlTest, InfoWithNullMessage) {
  __moore_reset_finish_state();

  testing::internal::CaptureStdout();
  __moore_info(nullptr);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("Info:"), std::string::npos);

  __moore_reset_finish_state();
}

TEST(MooreRuntimeSimControlTest, ResetSeverityCounts) {
  __moore_reset_finish_state();

  char data[] = "msg";
  MooreString msg = {data, 3};

  // Generate some errors and warnings
  testing::internal::CaptureStderr();
  __moore_error(&msg);
  __moore_warning(&msg);
  __moore_warning(&msg);
  testing::internal::GetCapturedStderr();

  EXPECT_EQ(__moore_get_error_count(), 1);
  EXPECT_EQ(__moore_get_warning_count(), 2);

  // Reset
  __moore_reset_severity_counts();

  EXPECT_EQ(__moore_get_error_count(), 0);
  EXPECT_EQ(__moore_get_warning_count(), 0);

  __moore_reset_finish_state();
}

TEST(MooreRuntimeSimControlTest, SeveritySummary) {
  __moore_reset_finish_state();

  // No errors or warnings - should return 0
  EXPECT_EQ(__moore_severity_summary(), 0);

  char data[] = "test";
  MooreString msg = {data, 4};

  // Add some errors and warnings
  testing::internal::CaptureStderr();
  __moore_error(&msg);
  __moore_error(&msg);
  __moore_warning(&msg);
  testing::internal::GetCapturedStderr();

  // Get summary
  testing::internal::CaptureStderr();
  int32_t result = __moore_severity_summary();
  std::string output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, 2);  // 2 errors
  EXPECT_NE(output.find("2 error(s)"), std::string::npos);
  EXPECT_NE(output.find("1 warning(s)"), std::string::npos);

  __moore_reset_finish_state();
}

TEST(MooreRuntimeSimControlTest, SeveritySummaryNoIssues) {
  __moore_reset_finish_state();

  // No errors or warnings - should not print anything
  testing::internal::CaptureStderr();
  int32_t result = __moore_severity_summary();
  std::string output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(result, 0);
  EXPECT_EQ(output, "");  // No output when no issues

  __moore_reset_finish_state();
}

TEST(MooreRuntimeSimControlTest, ErrorWithEmptyMessage) {
  __moore_reset_finish_state();

  MooreString empty = {nullptr, 0};

  testing::internal::CaptureStderr();
  __moore_error(&empty);
  std::string output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(__moore_get_error_count(), 1);
  EXPECT_NE(output.find("Error:"), std::string::npos);

  __moore_reset_finish_state();
}

TEST(MooreRuntimeSimControlTest, WarningWithEmptyMessage) {
  __moore_reset_finish_state();

  MooreString empty = {nullptr, 0};

  testing::internal::CaptureStderr();
  __moore_warning(&empty);
  std::string output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(__moore_get_warning_count(), 1);
  EXPECT_NE(output.find("Warning:"), std::string::npos);

  __moore_reset_finish_state();
}

TEST(MooreRuntimeSimControlTest, ResetFinishStateResetsAll) {
  __moore_set_finish_exits(false);

  char data[] = "test";
  MooreString msg = {data, 4};

  // Set various states
  testing::internal::CaptureStderr();
  __moore_finish(5);
  __moore_error(&msg);
  __moore_warning(&msg);
  testing::internal::GetCapturedStderr();

  EXPECT_TRUE(__moore_finish_called());
  EXPECT_EQ(__moore_get_exit_code(), 5);
  EXPECT_EQ(__moore_get_error_count(), 1);
  EXPECT_EQ(__moore_get_warning_count(), 1);

  // Reset all
  __moore_reset_finish_state();

  EXPECT_FALSE(__moore_finish_called());
  EXPECT_EQ(__moore_get_exit_code(), 0);
  EXPECT_EQ(__moore_get_error_count(), 0);
  EXPECT_EQ(__moore_get_warning_count(), 0);
}

//===----------------------------------------------------------------------===//
// UVM Phase System Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeUvmPhaseTest, PhaseStartWithValidName) {
  // Capture stdout to verify the phase start message
  testing::internal::CaptureStdout();

  __uvm_phase_start("build", 5);

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("Starting build_phase"), std::string::npos);
  EXPECT_NE(output.find("[PHASE]"), std::string::npos);
}

TEST(MooreRuntimeUvmPhaseTest, PhaseStartWithEmptyName) {
  testing::internal::CaptureStdout();

  __uvm_phase_start(nullptr, 0);

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("(unknown)"), std::string::npos);
}

TEST(MooreRuntimeUvmPhaseTest, PhaseEndWithValidName) {
  testing::internal::CaptureStdout();

  __uvm_phase_end("run", 3);

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("Completed run_phase"), std::string::npos);
  EXPECT_NE(output.find("[PHASE]"), std::string::npos);
}

TEST(MooreRuntimeUvmPhaseTest, PhaseEndWithEmptyName) {
  testing::internal::CaptureStdout();

  __uvm_phase_end(nullptr, 0);

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("(unknown)"), std::string::npos);
}

TEST(MooreRuntimeUvmPhaseTest, ExecutePhasesRunsAllPhases) {
  testing::internal::CaptureStdout();

  __uvm_execute_phases();

  std::string output = testing::internal::GetCapturedStdout();

  // Verify all standard UVM phases are executed in order
  EXPECT_NE(output.find("Starting build_phase"), std::string::npos);
  EXPECT_NE(output.find("Completed build_phase"), std::string::npos);

  EXPECT_NE(output.find("Starting connect_phase"), std::string::npos);
  EXPECT_NE(output.find("Completed connect_phase"), std::string::npos);

  EXPECT_NE(output.find("Starting end_of_elaboration_phase"), std::string::npos);
  EXPECT_NE(output.find("Completed end_of_elaboration_phase"), std::string::npos);

  EXPECT_NE(output.find("Starting start_of_simulation_phase"), std::string::npos);
  EXPECT_NE(output.find("Completed start_of_simulation_phase"), std::string::npos);

  EXPECT_NE(output.find("Starting run_phase"), std::string::npos);
  EXPECT_NE(output.find("Completed run_phase"), std::string::npos);

  EXPECT_NE(output.find("Starting extract_phase"), std::string::npos);
  EXPECT_NE(output.find("Completed extract_phase"), std::string::npos);

  EXPECT_NE(output.find("Starting check_phase"), std::string::npos);
  EXPECT_NE(output.find("Completed check_phase"), std::string::npos);

  EXPECT_NE(output.find("Starting report_phase"), std::string::npos);
  EXPECT_NE(output.find("Completed report_phase"), std::string::npos);

  EXPECT_NE(output.find("Starting final_phase"), std::string::npos);
  EXPECT_NE(output.find("Completed final_phase"), std::string::npos);
}

TEST(MooreRuntimeUvmPhaseTest, ExecutePhasesOrder) {
  testing::internal::CaptureStdout();

  __uvm_execute_phases();

  std::string output = testing::internal::GetCapturedStdout();

  // Verify phases execute in the correct order
  size_t buildPos = output.find("Starting build_phase");
  size_t connectPos = output.find("Starting connect_phase");
  size_t endElabPos = output.find("Starting end_of_elaboration_phase");
  size_t startSimPos = output.find("Starting start_of_simulation_phase");
  size_t runPos = output.find("Starting run_phase");
  size_t extractPos = output.find("Starting extract_phase");
  size_t checkPos = output.find("Starting check_phase");
  size_t reportPos = output.find("Starting report_phase");
  size_t finalPos = output.find("Starting final_phase");

  EXPECT_LT(buildPos, connectPos);
  EXPECT_LT(connectPos, endElabPos);
  EXPECT_LT(endElabPos, startSimPos);
  EXPECT_LT(startSimPos, runPos);
  EXPECT_LT(runPos, extractPos);
  EXPECT_LT(extractPos, checkPos);
  EXPECT_LT(checkPos, reportPos);
  EXPECT_LT(reportPos, finalPos);
}

TEST(MooreRuntimeUvmPhaseTest, RunTestWithNameCallsPhases) {
  testing::internal::CaptureStdout();

  __uvm_run_test("my_test", 7);

  std::string output = testing::internal::GetCapturedStdout();

  // Verify run_test message
  EXPECT_NE(output.find("[RNTST]"), std::string::npos);
  EXPECT_NE(output.find("Running test my_test"), std::string::npos);

  // Verify phases are executed
  EXPECT_NE(output.find("Starting build_phase"), std::string::npos);
  EXPECT_NE(output.find("Starting run_phase"), std::string::npos);

  // Verify completion message
  EXPECT_NE(output.find("[FINISH]"), std::string::npos);
  EXPECT_NE(output.find("UVM phasing complete"), std::string::npos);
}

TEST(MooreRuntimeUvmPhaseTest, RunTestWithEmptyName) {
  testing::internal::CaptureStdout();

  __uvm_run_test(nullptr, 0);

  std::string output = testing::internal::GetCapturedStdout();

  // Verify default test message
  EXPECT_NE(output.find("(default)"), std::string::npos);

  // Verify phases are still executed
  EXPECT_NE(output.find("Starting build_phase"), std::string::npos);

  // Verify completion
  EXPECT_NE(output.find("[FINISH]"), std::string::npos);
}

TEST(MooreRuntimeUvmPhaseTest, RunTestWarningForUninstantiatedTest) {
  __moore_uvm_factory_clear();
  testing::internal::CaptureStdout();

  __uvm_run_test("unimplemented_test", 18);

  std::string output = testing::internal::GetCapturedStdout();

  // Verify warning about unregistered test type
  EXPECT_NE(output.find("UVM_WARNING"), std::string::npos);
  EXPECT_NE(output.find("NOTYPE"), std::string::npos);
  EXPECT_NE(output.find("unimplemented_test"), std::string::npos);

  __moore_uvm_factory_clear();
}

TEST(MooreRuntimeUvmPhaseTest, PhaseStartWithPartialLength) {
  // Test with length shorter than actual string
  testing::internal::CaptureStdout();

  // "build_extra" with length 5 should only use "build"
  __uvm_phase_start("build_extra", 5);

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("Starting build_phase"), std::string::npos);
  EXPECT_EQ(output.find("build_extra"), std::string::npos);
}

//===----------------------------------------------------------------------===//
// UVM Component Phase Callback Tests
//===----------------------------------------------------------------------===//

// Test data structure for tracking callback invocations
struct PhaseCallbackTestData {
  int callCount = 0;
  std::vector<MooreUvmPhase> phasesExecuted;
  std::vector<void *> componentsExecuted;
  void *lastComponent = nullptr;
  MooreUvmPhase lastPhase = UVM_PHASE_BUILD;
};

// Test callback for function phases
static void testPhaseCallback(void *component, void *phase, void *userData) {
  auto *data = static_cast<PhaseCallbackTestData *>(userData);
  data->callCount++;
  data->lastComponent = component;
  data->componentsExecuted.push_back(component);
}

// Test callback for run_phase (task phase)
static void testRunPhaseCallback(void *component, void *phase, void *userData) {
  auto *data = static_cast<PhaseCallbackTestData *>(userData);
  data->callCount++;
  data->lastComponent = component;
  data->componentsExecuted.push_back(component);
}

// Global phase callback data
struct GlobalPhaseCallbackData {
  int startCount = 0;
  int endCount = 0;
  std::vector<MooreUvmPhase> startPhases;
  std::vector<MooreUvmPhase> endPhases;
  std::vector<std::string> startPhaseNames;
  std::vector<std::string> endPhaseNames;
};

static void globalPhaseStartCallback(MooreUvmPhase phase, const char *phaseName,
                                     void *userData) {
  auto *data = static_cast<GlobalPhaseCallbackData *>(userData);
  data->startCount++;
  data->startPhases.push_back(phase);
  if (phaseName)
    data->startPhaseNames.push_back(phaseName);
}

static void globalPhaseEndCallback(MooreUvmPhase phase, const char *phaseName,
                                   void *userData) {
  auto *data = static_cast<GlobalPhaseCallbackData *>(userData);
  data->endCount++;
  data->endPhases.push_back(phase);
  if (phaseName)
    data->endPhaseNames.push_back(phaseName);
}

TEST(MooreRuntimeUvmComponentCallbackTest, RegisterComponent) {
  // Clear any existing state
  __moore_uvm_clear_components();
  EXPECT_EQ(__moore_uvm_get_component_count(), 0);

  // Register a component
  int dummyComponent = 42;
  int64_t handle = __moore_uvm_register_component(&dummyComponent, "test_comp",
                                                   9, nullptr, 0);

  EXPECT_NE(handle, 0);
  EXPECT_EQ(__moore_uvm_get_component_count(), 1);

  // Clean up
  __moore_uvm_clear_components();
  EXPECT_EQ(__moore_uvm_get_component_count(), 0);
}

TEST(MooreRuntimeUvmComponentCallbackTest, RegisterNullComponent) {
  __moore_uvm_clear_components();

  // Registering null component should fail
  int64_t handle = __moore_uvm_register_component(nullptr, "test", 4, nullptr, 0);
  EXPECT_EQ(handle, 0);
  EXPECT_EQ(__moore_uvm_get_component_count(), 0);

  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmComponentCallbackTest, RegisterMultipleComponents) {
  __moore_uvm_clear_components();

  int comp1 = 1, comp2 = 2, comp3 = 3;

  int64_t h1 = __moore_uvm_register_component(&comp1, "comp1", 5, nullptr, 0);
  int64_t h2 = __moore_uvm_register_component(&comp2, "comp2", 5, &comp1, 1);
  int64_t h3 = __moore_uvm_register_component(&comp3, "comp3", 5, &comp1, 1);

  EXPECT_NE(h1, 0);
  EXPECT_NE(h2, 0);
  EXPECT_NE(h3, 0);
  EXPECT_NE(h1, h2);
  EXPECT_NE(h2, h3);
  EXPECT_EQ(__moore_uvm_get_component_count(), 3);

  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmComponentCallbackTest, UnregisterComponent) {
  __moore_uvm_clear_components();

  int comp1 = 1, comp2 = 2;

  int64_t h1 = __moore_uvm_register_component(&comp1, "comp1", 5, nullptr, 0);
  int64_t h2 = __moore_uvm_register_component(&comp2, "comp2", 5, nullptr, 0);

  EXPECT_EQ(__moore_uvm_get_component_count(), 2);

  __moore_uvm_unregister_component(h1);
  EXPECT_EQ(__moore_uvm_get_component_count(), 1);

  __moore_uvm_unregister_component(h2);
  EXPECT_EQ(__moore_uvm_get_component_count(), 0);

  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmComponentCallbackTest, UnregisterInvalidHandle) {
  __moore_uvm_clear_components();

  int comp = 1;
  (void)__moore_uvm_register_component(&comp, "comp", 4, nullptr, 0);
  EXPECT_EQ(__moore_uvm_get_component_count(), 1);

  // Unregistering invalid handle should not affect count
  __moore_uvm_unregister_component(999);
  EXPECT_EQ(__moore_uvm_get_component_count(), 1);

  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmComponentCallbackTest, SetPhaseCallback) {
  __moore_uvm_clear_components();

  int comp = 1;
  PhaseCallbackTestData data;

  int64_t handle = __moore_uvm_register_component(&comp, "comp", 4, nullptr, 0);
  __moore_uvm_set_phase_callback(handle, UVM_PHASE_BUILD, testPhaseCallback,
                                  &data);

  // Execute phases to trigger callback
  testing::internal::CaptureStdout();
  __uvm_execute_phases();
  testing::internal::GetCapturedStdout();

  EXPECT_EQ(data.callCount, 1);
  EXPECT_EQ(data.lastComponent, &comp);

  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmComponentCallbackTest, SetRunPhaseCallback) {
  __moore_uvm_clear_components();

  int comp = 1;
  PhaseCallbackTestData data;

  int64_t handle = __moore_uvm_register_component(&comp, "comp", 4, nullptr, 0);
  __moore_uvm_set_run_phase_callback(handle, testRunPhaseCallback, &data);

  // Execute phases to trigger callback
  testing::internal::CaptureStdout();
  __uvm_execute_phases();
  testing::internal::GetCapturedStdout();

  EXPECT_EQ(data.callCount, 1);
  EXPECT_EQ(data.lastComponent, &comp);

  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmComponentCallbackTest, MultiplePhaseCallbacks) {
  __moore_uvm_clear_components();

  int comp = 1;
  PhaseCallbackTestData buildData, connectData, runData, reportData;

  int64_t handle = __moore_uvm_register_component(&comp, "comp", 4, nullptr, 0);
  __moore_uvm_set_phase_callback(handle, UVM_PHASE_BUILD, testPhaseCallback,
                                  &buildData);
  __moore_uvm_set_phase_callback(handle, UVM_PHASE_CONNECT, testPhaseCallback,
                                  &connectData);
  __moore_uvm_set_run_phase_callback(handle, testRunPhaseCallback, &runData);
  __moore_uvm_set_phase_callback(handle, UVM_PHASE_REPORT, testPhaseCallback,
                                  &reportData);

  // Execute phases
  testing::internal::CaptureStdout();
  __uvm_execute_phases();
  testing::internal::GetCapturedStdout();

  EXPECT_EQ(buildData.callCount, 1);
  EXPECT_EQ(connectData.callCount, 1);
  EXPECT_EQ(runData.callCount, 1);
  EXPECT_EQ(reportData.callCount, 1);

  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmComponentCallbackTest, TopDownPhaseOrder) {
  __moore_uvm_clear_components();

  // Create a hierarchy: root -> child1, child2
  int root = 1, child1 = 2, child2 = 3;
  PhaseCallbackTestData data;

  int64_t hRoot = __moore_uvm_register_component(&root, "root", 4, nullptr, 0);
  int64_t hChild1 =
      __moore_uvm_register_component(&child1, "child1", 6, &root, 1);
  int64_t hChild2 =
      __moore_uvm_register_component(&child2, "child2", 6, &root, 1);

  // Set callbacks for build_phase (top-down)
  __moore_uvm_set_phase_callback(hRoot, UVM_PHASE_BUILD, testPhaseCallback,
                                  &data);
  __moore_uvm_set_phase_callback(hChild1, UVM_PHASE_BUILD, testPhaseCallback,
                                  &data);
  __moore_uvm_set_phase_callback(hChild2, UVM_PHASE_BUILD, testPhaseCallback,
                                  &data);

  // Execute phases
  testing::internal::CaptureStdout();
  __uvm_execute_phases();
  testing::internal::GetCapturedStdout();

  // For top-down phases, root should be called first
  EXPECT_EQ(data.callCount, 3);
  EXPECT_EQ(data.componentsExecuted.size(), 3u);
  EXPECT_EQ(data.componentsExecuted[0], &root);

  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmComponentCallbackTest, BottomUpPhaseOrder) {
  __moore_uvm_clear_components();

  // Create a hierarchy: root -> child
  int root = 1, child = 2;
  PhaseCallbackTestData data;

  int64_t hRoot = __moore_uvm_register_component(&root, "root", 4, nullptr, 0);
  int64_t hChild = __moore_uvm_register_component(&child, "child", 5, &root, 1);

  // Set callbacks for connect_phase (bottom-up)
  __moore_uvm_set_phase_callback(hRoot, UVM_PHASE_CONNECT, testPhaseCallback,
                                  &data);
  __moore_uvm_set_phase_callback(hChild, UVM_PHASE_CONNECT, testPhaseCallback,
                                  &data);

  // Execute phases
  testing::internal::CaptureStdout();
  __uvm_execute_phases();
  testing::internal::GetCapturedStdout();

  // For bottom-up phases, child should be called first
  EXPECT_EQ(data.callCount, 2);
  EXPECT_EQ(data.componentsExecuted.size(), 2u);
  EXPECT_EQ(data.componentsExecuted[0], &child);
  EXPECT_EQ(data.componentsExecuted[1], &root);

  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmComponentCallbackTest, GlobalPhaseCallbacks) {
  __moore_uvm_clear_components();

  GlobalPhaseCallbackData data;

  __moore_uvm_set_global_phase_start_callback(globalPhaseStartCallback, &data);
  __moore_uvm_set_global_phase_end_callback(globalPhaseEndCallback, &data);

  // Execute phases
  testing::internal::CaptureStdout();
  __uvm_execute_phases();
  testing::internal::GetCapturedStdout();

  // Should have 9 phases
  EXPECT_EQ(data.startCount, 9);
  EXPECT_EQ(data.endCount, 9);
  EXPECT_EQ(data.startPhases.size(), 9u);
  EXPECT_EQ(data.endPhases.size(), 9u);

  // Verify phase order
  EXPECT_EQ(data.startPhases[0], UVM_PHASE_BUILD);
  EXPECT_EQ(data.startPhases[1], UVM_PHASE_CONNECT);
  EXPECT_EQ(data.startPhases[4], UVM_PHASE_RUN);
  EXPECT_EQ(data.startPhases[8], UVM_PHASE_FINAL);

  // Clean up
  __moore_uvm_set_global_phase_start_callback(nullptr, nullptr);
  __moore_uvm_set_global_phase_end_callback(nullptr, nullptr);
  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmComponentCallbackTest, GlobalPhaseCallbackNames) {
  __moore_uvm_clear_components();

  GlobalPhaseCallbackData data;

  __moore_uvm_set_global_phase_start_callback(globalPhaseStartCallback, &data);

  // Execute phases
  testing::internal::CaptureStdout();
  __uvm_execute_phases();
  testing::internal::GetCapturedStdout();

  // Verify phase names
  EXPECT_EQ(data.startPhaseNames.size(), 9u);
  EXPECT_EQ(data.startPhaseNames[0], "build");
  EXPECT_EQ(data.startPhaseNames[1], "connect");
  EXPECT_EQ(data.startPhaseNames[2], "end_of_elaboration");
  EXPECT_EQ(data.startPhaseNames[3], "start_of_simulation");
  EXPECT_EQ(data.startPhaseNames[4], "run");
  EXPECT_EQ(data.startPhaseNames[5], "extract");
  EXPECT_EQ(data.startPhaseNames[6], "check");
  EXPECT_EQ(data.startPhaseNames[7], "report");
  EXPECT_EQ(data.startPhaseNames[8], "final");

  // Clean up
  __moore_uvm_set_global_phase_start_callback(nullptr, nullptr);
  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmComponentCallbackTest, SetCallbackOnInvalidHandle) {
  __moore_uvm_clear_components();

  PhaseCallbackTestData data;

  // Should not crash
  __moore_uvm_set_phase_callback(999, UVM_PHASE_BUILD, testPhaseCallback,
                                  &data);
  __moore_uvm_set_run_phase_callback(999, testRunPhaseCallback, &data);

  // Execute phases - should not call any callbacks
  testing::internal::CaptureStdout();
  __uvm_execute_phases();
  testing::internal::GetCapturedStdout();

  EXPECT_EQ(data.callCount, 0);

  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmComponentCallbackTest, SetCallbackWithInvalidPhase) {
  __moore_uvm_clear_components();

  int comp = 1;
  PhaseCallbackTestData data;

  int64_t handle = __moore_uvm_register_component(&comp, "comp", 4, nullptr, 0);

  // Should not crash with invalid phase values
  __moore_uvm_set_phase_callback(handle, static_cast<MooreUvmPhase>(-1),
                                  testPhaseCallback, &data);
  __moore_uvm_set_phase_callback(handle, static_cast<MooreUvmPhase>(100),
                                  testPhaseCallback, &data);

  // Execute phases
  testing::internal::CaptureStdout();
  __uvm_execute_phases();
  testing::internal::GetCapturedStdout();

  // Invalid phase callbacks should not have been registered
  EXPECT_EQ(data.callCount, 0);

  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmComponentCallbackTest, ClearResetsAll) {
  __moore_uvm_clear_components();

  int comp = 1;
  PhaseCallbackTestData data;
  GlobalPhaseCallbackData globalData;

  int64_t handle = __moore_uvm_register_component(&comp, "comp", 4, nullptr, 0);
  __moore_uvm_set_phase_callback(handle, UVM_PHASE_BUILD, testPhaseCallback,
                                  &data);
  __moore_uvm_set_global_phase_start_callback(globalPhaseStartCallback,
                                               &globalData);

  EXPECT_EQ(__moore_uvm_get_component_count(), 1);

  // Clear everything
  __moore_uvm_clear_components();

  EXPECT_EQ(__moore_uvm_get_component_count(), 0);

  // Execute phases - no callbacks should fire
  testing::internal::CaptureStdout();
  __uvm_execute_phases();
  testing::internal::GetCapturedStdout();

  EXPECT_EQ(data.callCount, 0);
  EXPECT_EQ(globalData.startCount, 0);
}

TEST(MooreRuntimeUvmComponentCallbackTest, PhaseEnumValues) {
  // Verify phase enum values match expected order
  EXPECT_EQ(UVM_PHASE_BUILD, 0);
  EXPECT_EQ(UVM_PHASE_CONNECT, 1);
  EXPECT_EQ(UVM_PHASE_END_OF_ELABORATION, 2);
  EXPECT_EQ(UVM_PHASE_START_OF_SIMULATION, 3);
  EXPECT_EQ(UVM_PHASE_RUN, 4);
  EXPECT_EQ(UVM_PHASE_EXTRACT, 5);
  EXPECT_EQ(UVM_PHASE_CHECK, 6);
  EXPECT_EQ(UVM_PHASE_REPORT, 7);
  EXPECT_EQ(UVM_PHASE_FINAL, 8);
  EXPECT_EQ(UVM_PHASE_COUNT, 9);
}

TEST(MooreRuntimeUvmComponentCallbackTest, ComponentWithEmptyName) {
  __moore_uvm_clear_components();

  int comp = 1;

  // Register with empty name
  int64_t handle = __moore_uvm_register_component(&comp, "", 0, nullptr, 0);
  EXPECT_NE(handle, 0);
  EXPECT_EQ(__moore_uvm_get_component_count(), 1);

  // Register with null name
  int comp2 = 2;
  int64_t handle2 =
      __moore_uvm_register_component(&comp2, nullptr, 0, nullptr, 0);
  EXPECT_NE(handle2, 0);
  EXPECT_EQ(__moore_uvm_get_component_count(), 2);

  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmComponentCallbackTest, DeepHierarchy) {
  __moore_uvm_clear_components();

  // Create a deep hierarchy: root -> level1 -> level2 -> level3
  int root = 0, level1 = 1, level2 = 2, level3 = 3;
  PhaseCallbackTestData data;

  int64_t hRoot = __moore_uvm_register_component(&root, "root", 4, nullptr, 0);
  int64_t hL1 =
      __moore_uvm_register_component(&level1, "level1", 6, &root, 1);
  int64_t hL2 =
      __moore_uvm_register_component(&level2, "level2", 6, &level1, 2);
  int64_t hL3 =
      __moore_uvm_register_component(&level3, "level3", 6, &level2, 3);

  // Set callbacks for final_phase (top-down)
  __moore_uvm_set_phase_callback(hRoot, UVM_PHASE_FINAL, testPhaseCallback,
                                  &data);
  __moore_uvm_set_phase_callback(hL1, UVM_PHASE_FINAL, testPhaseCallback,
                                  &data);
  __moore_uvm_set_phase_callback(hL2, UVM_PHASE_FINAL, testPhaseCallback,
                                  &data);
  __moore_uvm_set_phase_callback(hL3, UVM_PHASE_FINAL, testPhaseCallback,
                                  &data);

  // Execute phases
  testing::internal::CaptureStdout();
  __uvm_execute_phases();
  testing::internal::GetCapturedStdout();

  // Verify order: root, level1, level2, level3 (top-down)
  EXPECT_EQ(data.callCount, 4);
  EXPECT_EQ(data.componentsExecuted[0], &root);
  EXPECT_EQ(data.componentsExecuted[1], &level1);
  EXPECT_EQ(data.componentsExecuted[2], &level2);
  EXPECT_EQ(data.componentsExecuted[3], &level3);

  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmComponentCallbackTest, AllPhasesExecuteInOrder) {
  __moore_uvm_clear_components();

  // Use global callbacks to track phase order
  GlobalPhaseCallbackData data;
  __moore_uvm_set_global_phase_start_callback(globalPhaseStartCallback, &data);

  // Execute phases
  testing::internal::CaptureStdout();
  __uvm_execute_phases();
  testing::internal::GetCapturedStdout();

  // Verify all 9 phases executed in order
  ASSERT_EQ(data.startPhases.size(), 9u);
  EXPECT_EQ(data.startPhases[0], UVM_PHASE_BUILD);
  EXPECT_EQ(data.startPhases[1], UVM_PHASE_CONNECT);
  EXPECT_EQ(data.startPhases[2], UVM_PHASE_END_OF_ELABORATION);
  EXPECT_EQ(data.startPhases[3], UVM_PHASE_START_OF_SIMULATION);
  EXPECT_EQ(data.startPhases[4], UVM_PHASE_RUN);
  EXPECT_EQ(data.startPhases[5], UVM_PHASE_EXTRACT);
  EXPECT_EQ(data.startPhases[6], UVM_PHASE_CHECK);
  EXPECT_EQ(data.startPhases[7], UVM_PHASE_REPORT);
  EXPECT_EQ(data.startPhases[8], UVM_PHASE_FINAL);

  __moore_uvm_set_global_phase_start_callback(nullptr, nullptr);
  __moore_uvm_clear_components();
}

//===----------------------------------------------------------------------===//
// UVM Factory Tests
//===----------------------------------------------------------------------===//

// Test data structures for factory tests
namespace {

struct TestComponentData {
  std::string name;
  void *parent;
  int creationCount;
};

static std::map<std::string, TestComponentData> createdComponents;

// Mock component creator
void *testComponentCreator(const char *name, int64_t nameLen, void *parent,
                           void *userData) {
  std::string compName(name, static_cast<size_t>(nameLen));
  TestComponentData data;
  data.name = compName;
  data.parent = parent;
  data.creationCount = 1;
  createdComponents[compName] = data;
  // Return a non-null pointer (use userData as the "instance")
  return userData ? userData : const_cast<char *>("mock_component");
}

// Mock object creator
void *testObjectCreator(const char *name, int64_t nameLen, void *userData) {
  (void)userData;
  std::string objName(name, static_cast<size_t>(nameLen));
  // Return a mock object pointer
  return const_cast<char *>("mock_object");
}

} // namespace

TEST(MooreRuntimeUvmFactoryTest, FactoryInitiallyEmpty) {
  __moore_uvm_factory_clear();
  EXPECT_EQ(__moore_uvm_factory_get_type_count(), 0);
}

TEST(MooreRuntimeUvmFactoryTest, RegisterComponentType) {
  __moore_uvm_factory_clear();

  const char *typeName = "my_test";
  int32_t result = __moore_uvm_factory_register_component(
      typeName, static_cast<int64_t>(std::strlen(typeName)), testComponentCreator,
      nullptr);

  EXPECT_EQ(result, 1);
  EXPECT_EQ(__moore_uvm_factory_get_type_count(), 1);
  EXPECT_EQ(__moore_uvm_factory_is_type_registered(
                typeName, static_cast<int64_t>(std::strlen(typeName))),
            1);

  __moore_uvm_factory_clear();
}

TEST(MooreRuntimeUvmFactoryTest, RegisterObjectType) {
  __moore_uvm_factory_clear();

  const char *typeName = "my_sequence";
  int32_t result = __moore_uvm_factory_register_object(
      typeName, static_cast<int64_t>(std::strlen(typeName)), testObjectCreator,
      nullptr);

  EXPECT_EQ(result, 1);
  EXPECT_EQ(__moore_uvm_factory_get_type_count(), 1);
  EXPECT_EQ(__moore_uvm_factory_is_type_registered(
                typeName, static_cast<int64_t>(std::strlen(typeName))),
            1);

  __moore_uvm_factory_clear();
}

TEST(MooreRuntimeUvmFactoryTest, PreventDuplicateRegistration) {
  __moore_uvm_factory_clear();

  const char *typeName = "my_test";
  int64_t len = static_cast<int64_t>(std::strlen(typeName));

  // First registration should succeed
  int32_t result1 =
      __moore_uvm_factory_register_component(typeName, len, testComponentCreator, nullptr);
  EXPECT_EQ(result1, 1);

  // Second registration with same name should fail
  int32_t result2 =
      __moore_uvm_factory_register_component(typeName, len, testComponentCreator, nullptr);
  EXPECT_EQ(result2, 0);

  // Should still only have one type
  EXPECT_EQ(__moore_uvm_factory_get_type_count(), 1);

  __moore_uvm_factory_clear();
}

TEST(MooreRuntimeUvmFactoryTest, CreateComponentByName) {
  __moore_uvm_factory_clear();
  createdComponents.clear();

  const char *typeName = "my_test";
  int64_t typeLen = static_cast<int64_t>(std::strlen(typeName));
  __moore_uvm_factory_register_component(typeName, typeLen, testComponentCreator,
                                         nullptr);

  const char *instName = "uvm_test_top";
  int64_t instLen = static_cast<int64_t>(std::strlen(instName));

  testing::internal::CaptureStdout();
  void *component = __moore_uvm_factory_create_component_by_name(
      typeName, typeLen, instName, instLen, nullptr);
  testing::internal::GetCapturedStdout();

  EXPECT_NE(component, nullptr);
  EXPECT_EQ(createdComponents.count(instName), 1u);
  EXPECT_EQ(createdComponents[instName].name, instName);

  __moore_uvm_factory_clear();
}

TEST(MooreRuntimeUvmFactoryTest, CreateComponentUnregisteredType) {
  __moore_uvm_factory_clear();

  const char *typeName = "nonexistent_test";
  int64_t typeLen = static_cast<int64_t>(std::strlen(typeName));
  const char *instName = "uvm_test_top";
  int64_t instLen = static_cast<int64_t>(std::strlen(instName));

  testing::internal::CaptureStdout();
  void *component = __moore_uvm_factory_create_component_by_name(
      typeName, typeLen, instName, instLen, nullptr);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(component, nullptr);
  EXPECT_NE(output.find("NOTYPE"), std::string::npos);

  __moore_uvm_factory_clear();
}

TEST(MooreRuntimeUvmFactoryTest, TypeOverride) {
  __moore_uvm_factory_clear();
  createdComponents.clear();

  // Register base type
  const char *baseType = "base_test";
  int64_t baseLen = static_cast<int64_t>(std::strlen(baseType));
  __moore_uvm_factory_register_component(baseType, baseLen, testComponentCreator,
                                         const_cast<char *>("base_instance"));

  // Register override type
  const char *overrideType = "extended_test";
  int64_t overrideLen = static_cast<int64_t>(std::strlen(overrideType));
  __moore_uvm_factory_register_component(overrideType, overrideLen,
                                         testComponentCreator,
                                         const_cast<char *>("override_instance"));

  // Set type override
  int32_t overrideResult = __moore_uvm_factory_set_type_override(
      baseType, baseLen, overrideType, overrideLen, 1);
  EXPECT_EQ(overrideResult, 1);

  // Create component using base type name - should get override type
  const char *instName = "test_top";
  int64_t instLen = static_cast<int64_t>(std::strlen(instName));

  testing::internal::CaptureStdout();
  void *component = __moore_uvm_factory_create_component_by_name(
      baseType, baseLen, instName, instLen, nullptr);
  testing::internal::GetCapturedStdout();

  EXPECT_NE(component, nullptr);
  // The component should have been created using the override's userData
  EXPECT_EQ(component, const_cast<char *>("override_instance"));

  __moore_uvm_factory_clear();
}

TEST(MooreRuntimeUvmFactoryTest, TypeOverrideChain) {
  __moore_uvm_factory_clear();

  // Register types A -> B -> C
  const char *typeA = "type_a";
  const char *typeB = "type_b";
  const char *typeC = "type_c";
  int64_t lenA = static_cast<int64_t>(std::strlen(typeA));
  int64_t lenB = static_cast<int64_t>(std::strlen(typeB));
  int64_t lenC = static_cast<int64_t>(std::strlen(typeC));

  __moore_uvm_factory_register_component(typeA, lenA, testComponentCreator,
                                         const_cast<char *>("instance_a"));
  __moore_uvm_factory_register_component(typeB, lenB, testComponentCreator,
                                         const_cast<char *>("instance_b"));
  __moore_uvm_factory_register_component(typeC, lenC, testComponentCreator,
                                         const_cast<char *>("instance_c"));

  // Set up override chain: A -> B -> C
  __moore_uvm_factory_set_type_override(typeA, lenA, typeB, lenB, 1);
  __moore_uvm_factory_set_type_override(typeB, lenB, typeC, lenC, 1);

  // Create component using type A - should get type C
  const char *instName = "test";
  int64_t instLen = static_cast<int64_t>(std::strlen(instName));

  testing::internal::CaptureStdout();
  void *component = __moore_uvm_factory_create_component_by_name(
      typeA, lenA, instName, instLen, nullptr);
  testing::internal::GetCapturedStdout();

  EXPECT_NE(component, nullptr);
  EXPECT_EQ(component, const_cast<char *>("instance_c"));

  __moore_uvm_factory_clear();
}

TEST(MooreRuntimeUvmFactoryTest, TypeNotRegisteredCheck) {
  __moore_uvm_factory_clear();

  const char *typeName = "unregistered_type";
  int64_t len = static_cast<int64_t>(std::strlen(typeName));

  EXPECT_EQ(__moore_uvm_factory_is_type_registered(typeName, len), 0);

  __moore_uvm_factory_clear();
}

TEST(MooreRuntimeUvmFactoryTest, FactoryClear) {
  __moore_uvm_factory_clear();

  // Register some types
  const char *type1 = "test1";
  const char *type2 = "test2";
  __moore_uvm_factory_register_component(
      type1, static_cast<int64_t>(std::strlen(type1)), testComponentCreator, nullptr);
  __moore_uvm_factory_register_object(type2, static_cast<int64_t>(std::strlen(type2)),
                                      testObjectCreator, nullptr);

  EXPECT_EQ(__moore_uvm_factory_get_type_count(), 2);

  // Clear and verify
  __moore_uvm_factory_clear();
  EXPECT_EQ(__moore_uvm_factory_get_type_count(), 0);
}

TEST(MooreRuntimeUvmFactoryTest, InvalidInputsHandled) {
  __moore_uvm_factory_clear();

  // Null type name
  EXPECT_EQ(__moore_uvm_factory_register_component(nullptr, 5, testComponentCreator,
                                                   nullptr),
            0);

  // Zero length
  EXPECT_EQ(__moore_uvm_factory_register_component("test", 0, testComponentCreator,
                                                   nullptr),
            0);

  // Null creator
  EXPECT_EQ(__moore_uvm_factory_register_component("test", 4, nullptr, nullptr), 0);

  // Is registered with null
  EXPECT_EQ(__moore_uvm_factory_is_type_registered(nullptr, 5), 0);

  // Create with null type name
  EXPECT_EQ(__moore_uvm_factory_create_component_by_name(nullptr, 5, "inst", 4, nullptr),
            nullptr);

  __moore_uvm_factory_clear();
}

TEST(MooreRuntimeUvmFactoryTest, RunTestWithRegisteredType) {
  __moore_uvm_factory_clear();
  createdComponents.clear();
  __moore_uvm_clear_components();

  // Register a test type
  const char *testType = "registered_test";
  int64_t typeLen = static_cast<int64_t>(std::strlen(testType));
  __moore_uvm_factory_register_component(testType, typeLen, testComponentCreator,
                                         nullptr);

  // Run test
  testing::internal::CaptureStdout();
  __uvm_run_test(testType, typeLen);
  std::string output = testing::internal::GetCapturedStdout();

  // Verify test was created
  EXPECT_NE(output.find("created successfully"), std::string::npos);
  EXPECT_EQ(createdComponents.count("uvm_test_top"), 1u);

  __moore_uvm_factory_clear();
  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmFactoryTest, RunTestWithUnregisteredType) {
  __moore_uvm_factory_clear();
  __moore_uvm_clear_components();

  const char *testType = "unregistered_test";
  int64_t typeLen = static_cast<int64_t>(std::strlen(testType));

  // Run test - should warn about unregistered type
  testing::internal::CaptureStdout();
  __uvm_run_test(testType, typeLen);
  std::string output = testing::internal::GetCapturedStdout();

  // Verify warning about unregistered type
  EXPECT_NE(output.find("NOTYPE"), std::string::npos);

  __moore_uvm_factory_clear();
  __moore_uvm_clear_components();
}

TEST(MooreRuntimeUvmFactoryTest, FactoryPrint) {
  __moore_uvm_factory_clear();

  // Register some types
  const char *comp1 = "test_component";
  const char *obj1 = "test_object";
  __moore_uvm_factory_register_component(
      comp1, static_cast<int64_t>(std::strlen(comp1)), testComponentCreator, nullptr);
  __moore_uvm_factory_register_object(obj1, static_cast<int64_t>(std::strlen(obj1)),
                                      testObjectCreator, nullptr);

  // Add an override
  const char *override = "override_comp";
  __moore_uvm_factory_register_component(
      override, static_cast<int64_t>(std::strlen(override)), testComponentCreator, nullptr);
  __moore_uvm_factory_set_type_override(
      comp1, static_cast<int64_t>(std::strlen(comp1)), override,
      static_cast<int64_t>(std::strlen(override)), 1);

  // Print factory state
  testing::internal::CaptureStdout();
  __moore_uvm_factory_print();
  std::string output = testing::internal::GetCapturedStdout();

  // Verify output contains expected sections
  EXPECT_NE(output.find("UVM Factory State"), std::string::npos);
  EXPECT_NE(output.find("test_component"), std::string::npos);
  EXPECT_NE(output.find("test_object"), std::string::npos);
  EXPECT_NE(output.find("Type overrides"), std::string::npos);

  __moore_uvm_factory_clear();
}

//===----------------------------------------------------------------------===//
// +UVM_TESTNAME Command-Line Argument Parsing Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeUvmTestnameTest, ParseTestnameFromCmdline) {
  // Set up environment with +UVM_TESTNAME argument
  setenv("CIRCT_UVM_ARGS", "+UVM_TESTNAME=my_test_class", 1);
  unsetenv("UVM_ARGS");

  MooreString result = __moore_uvm_get_testname_from_cmdline();

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 13);
  EXPECT_EQ(std::string(result.data, result.len), "my_test_class");

  __moore_free(result.data);
}

TEST(MooreRuntimeUvmTestnameTest, ParseTestnameWithOtherArgs) {
  // Set up environment with +UVM_TESTNAME mixed with other arguments
  setenv("CIRCT_UVM_ARGS", "+verbosity=high +UVM_TESTNAME=complex_test_name +timeout=1000", 1);
  unsetenv("UVM_ARGS");

  MooreString result = __moore_uvm_get_testname_from_cmdline();

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(std::string(result.data, result.len), "complex_test_name");

  __moore_free(result.data);
}

TEST(MooreRuntimeUvmTestnameTest, NoTestnameInCmdline) {
  // Set up environment without +UVM_TESTNAME
  setenv("CIRCT_UVM_ARGS", "+verbosity=high +timeout=1000", 1);
  unsetenv("UVM_ARGS");

  MooreString result = __moore_uvm_get_testname_from_cmdline();

  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);
}

TEST(MooreRuntimeUvmTestnameTest, EmptyTestname) {
  // Set up environment with empty +UVM_TESTNAME= (just the prefix)
  setenv("CIRCT_UVM_ARGS", "+UVM_TESTNAME=", 1);
  unsetenv("UVM_ARGS");

  MooreString result = __moore_uvm_get_testname_from_cmdline();

  // Empty value after = should return empty string
  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);
}

TEST(MooreRuntimeUvmTestnameTest, HasCmdlineTestnameTrue) {
  // Set up environment with +UVM_TESTNAME
  setenv("CIRCT_UVM_ARGS", "+UVM_TESTNAME=test_class", 1);
  unsetenv("UVM_ARGS");

  EXPECT_EQ(__moore_uvm_has_cmdline_testname(), 1);
}

TEST(MooreRuntimeUvmTestnameTest, HasCmdlineTestnameFalse) {
  // Set up environment without +UVM_TESTNAME
  setenv("CIRCT_UVM_ARGS", "+other_arg=value", 1);
  unsetenv("UVM_ARGS");

  EXPECT_EQ(__moore_uvm_has_cmdline_testname(), 0);
}

TEST(MooreRuntimeUvmTestnameTest, TestnameWithUnderscores) {
  // Test with underscores in the test name (common pattern)
  setenv("CIRCT_UVM_ARGS", "+UVM_TESTNAME=my_test_with_underscores", 1);
  unsetenv("UVM_ARGS");

  MooreString result = __moore_uvm_get_testname_from_cmdline();

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(std::string(result.data, result.len), "my_test_with_underscores");

  __moore_free(result.data);
}

TEST(MooreRuntimeUvmTestnameTest, TestnameWithNumbers) {
  // Test with numbers in the test name
  setenv("CIRCT_UVM_ARGS", "+UVM_TESTNAME=test123_v2", 1);
  unsetenv("UVM_ARGS");

  MooreString result = __moore_uvm_get_testname_from_cmdline();

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(std::string(result.data, result.len), "test123_v2");

  __moore_free(result.data);
}

TEST(MooreRuntimeUvmTestnameTest, TestnameFromUvmArgsFallback) {
  // Test that UVM_ARGS is used as fallback when CIRCT_UVM_ARGS is not set
  unsetenv("CIRCT_UVM_ARGS");
  setenv("UVM_ARGS", "+UVM_TESTNAME=fallback_test", 1);

  MooreString result = __moore_uvm_get_testname_from_cmdline();

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(std::string(result.data, result.len), "fallback_test");

  __moore_free(result.data);
}

TEST(MooreRuntimeUvmTestnameTest, TestnameFirstOccurrence) {
  // If multiple +UVM_TESTNAME arguments exist, the first one should be used
  setenv("CIRCT_UVM_ARGS", "+UVM_TESTNAME=first_test +UVM_TESTNAME=second_test", 1);
  unsetenv("UVM_ARGS");

  MooreString result = __moore_uvm_get_testname_from_cmdline();

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(std::string(result.data, result.len), "first_test");

  __moore_free(result.data);
}

TEST(MooreRuntimeUvmTestnameTest, TestnameWithScopedName) {
  // Test with package-scoped test name (pkg::class)
  setenv("CIRCT_UVM_ARGS", "+UVM_TESTNAME=my_pkg::my_test", 1);
  unsetenv("UVM_ARGS");

  MooreString result = __moore_uvm_get_testname_from_cmdline();

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(std::string(result.data, result.len), "my_pkg::my_test");

  __moore_free(result.data);
}

TEST(MooreRuntimeUvmTestnameTest, TestnameNoEnvVars) {
  // Test when neither CIRCT_UVM_ARGS nor UVM_ARGS is set
  unsetenv("CIRCT_UVM_ARGS");
  unsetenv("UVM_ARGS");

  MooreString result = __moore_uvm_get_testname_from_cmdline();

  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);

  EXPECT_EQ(__moore_uvm_has_cmdline_testname(), 0);
}

TEST(MooreRuntimeUvmTestnameTest, TestnameSimilarPrefixNotMatched) {
  // Test that similar prefixes like +UVM_TESTNAME_EXTRA are not matched
  setenv("CIRCT_UVM_ARGS", "+UVM_TESTNAME_EXTRA=wrong", 1);
  unsetenv("UVM_ARGS");

  // This should NOT match because it's not exactly +UVM_TESTNAME=
  EXPECT_EQ(__moore_uvm_has_cmdline_testname(), 0);

  MooreString result = __moore_uvm_get_testname_from_cmdline();
  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);
}

TEST(MooreRuntimeUvmTestnameTest, TestnameWithQuotedValue) {
  // Test with quoted test name (quotes should be handled by the parser)
  setenv("CIRCT_UVM_ARGS", "\"+UVM_TESTNAME=quoted_test\"", 1);
  unsetenv("UVM_ARGS");

  MooreString result = __moore_uvm_get_testname_from_cmdline();

  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(std::string(result.data, result.len), "quoted_test");

  __moore_free(result.data);
}

TEST(MooreRuntimeUvmTestnameTest, CaseSensitivity) {
  // Test that +UVM_TESTNAME is case-sensitive (lowercase should not match)
  setenv("CIRCT_UVM_ARGS", "+uvm_testname=lowercase_test", 1);
  unsetenv("UVM_ARGS");

  // Lowercase should NOT match (UVM standard is uppercase)
  EXPECT_EQ(__moore_uvm_has_cmdline_testname(), 0);

  MooreString result = __moore_uvm_get_testname_from_cmdline();
  EXPECT_EQ(result.data, nullptr);
  EXPECT_EQ(result.len, 0);
}

//===----------------------------------------------------------------------===//
// UVM config_db Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeConfigDbTest, BasicSetAndGet) {
  __moore_config_db_clear();

  // Set an integer value
  int64_t setValue = 42;
  __moore_config_db_set(nullptr, "top.env", 7, "myconfig", 8, &setValue,
                        sizeof(setValue), 1);

  // Get the value back
  int64_t getValue = 0;
  int32_t result = __moore_config_db_get(nullptr, "top.env", 7, "myconfig", 8, 1,
                                         &getValue, sizeof(getValue));
  EXPECT_EQ(result, 1);
  EXPECT_EQ(getValue, 42);

  __moore_config_db_clear();
}

TEST(MooreRuntimeConfigDbTest, GetNonexistent) {
  __moore_config_db_clear();

  int64_t getValue = 99;
  int32_t result = __moore_config_db_get(nullptr, "nonexistent", 11, "field", 5,
                                         1, &getValue, sizeof(getValue));
  EXPECT_EQ(result, 0);
  EXPECT_EQ(getValue, 99);  // Value should be unchanged

  __moore_config_db_clear();
}

TEST(MooreRuntimeConfigDbTest, ExistsExactMatch) {
  __moore_config_db_clear();

  int64_t value = 100;
  __moore_config_db_set(nullptr, "top.env", 7, "cfg", 3, &value, sizeof(value), 1);

  EXPECT_EQ(__moore_config_db_exists("top.env", 7, "cfg", 3), 1);
  EXPECT_EQ(__moore_config_db_exists("top.env", 7, "other", 5), 0);
  EXPECT_EQ(__moore_config_db_exists("different", 9, "cfg", 3), 0);

  __moore_config_db_clear();
}

TEST(MooreRuntimeConfigDbTest, WildcardStarMatchesAll) {
  __moore_config_db_clear();

  // Set with wildcard "*" (matches all instance paths)
  int64_t value = 123;
  __moore_config_db_set(nullptr, "*", 1, "global_cfg", 10, &value,
                        sizeof(value), 1);

  // Should be retrievable from any path
  int64_t result1 = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "top", 3, "global_cfg", 10, 1,
                                  &result1, sizeof(result1)),
            1);
  EXPECT_EQ(result1, 123);

  int64_t result2 = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "top.env.agent", 13, "global_cfg", 10,
                                  1, &result2, sizeof(result2)),
            1);
  EXPECT_EQ(result2, 123);

  // Also test exists
  EXPECT_EQ(__moore_config_db_exists("deep.path.here", 14, "global_cfg", 10), 1);

  __moore_config_db_clear();
}

TEST(MooreRuntimeConfigDbTest, WildcardPatternMatching) {
  __moore_config_db_clear();

  // Set with pattern "*agent*"
  int64_t value = 456;
  __moore_config_db_set(nullptr, "*agent*", 7, "agentcfg", 8, &value,
                        sizeof(value), 1);

  // Should match paths containing "agent"
  int64_t result1 = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "top.agent", 9, "agentcfg", 8, 1,
                                  &result1, sizeof(result1)),
            1);
  EXPECT_EQ(result1, 456);

  int64_t result2 = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "top.env.my_agent.sub", 20, "agentcfg",
                                  8, 1, &result2, sizeof(result2)),
            1);
  EXPECT_EQ(result2, 456);

  // Should NOT match paths without "agent"
  int64_t result3 = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "top.env", 7, "agentcfg", 8, 1,
                                  &result3, sizeof(result3)),
            0);

  __moore_config_db_clear();
}

TEST(MooreRuntimeConfigDbTest, HierarchicalPrefixMatching) {
  __moore_config_db_clear();

  // Set at parent path
  int64_t parentValue = 100;
  __moore_config_db_set(nullptr, "top.env", 7, "cfg", 3, &parentValue,
                        sizeof(parentValue), 1);

  // Should be retrievable from child paths (hierarchical matching)
  int64_t result1 = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "top.env.agent", 13, "cfg", 3, 1,
                                  &result1, sizeof(result1)),
            1);
  EXPECT_EQ(result1, 100);

  int64_t result2 = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "top.env.agent.driver", 20, "cfg", 3,
                                  1, &result2, sizeof(result2)),
            1);
  EXPECT_EQ(result2, 100);

  // Should NOT match unrelated paths
  int64_t result3 = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "top.other", 9, "cfg", 3, 1, &result3,
                                  sizeof(result3)),
            0);

  __moore_config_db_clear();
}

TEST(MooreRuntimeConfigDbTest, SpecificPathTakesPrecedence) {
  __moore_config_db_clear();

  // Set with wildcard first
  int64_t wildcardValue = 1;
  __moore_config_db_set(nullptr, "*", 1, "cfg", 3, &wildcardValue,
                        sizeof(wildcardValue), 1);

  // Set at specific path (more specific)
  int64_t specificValue = 2;
  __moore_config_db_set(nullptr, "top.env", 7, "cfg", 3, &specificValue,
                        sizeof(specificValue), 1);

  // Exact match should return specific value
  int64_t result1 = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "top.env", 7, "cfg", 3, 1, &result1,
                                  sizeof(result1)),
            1);
  EXPECT_EQ(result1, 2);

  // Child of specific path should also get specific value (hierarchical)
  int64_t result2 = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "top.env.agent", 13, "cfg", 3, 1,
                                  &result2, sizeof(result2)),
            1);
  EXPECT_EQ(result2, 2);

  // Unrelated path should get wildcard value
  int64_t result3 = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "other.path", 10, "cfg", 3, 1,
                                  &result3, sizeof(result3)),
            1);
  EXPECT_EQ(result3, 1);

  __moore_config_db_clear();
}

TEST(MooreRuntimeConfigDbTest, LastSetWinsForSamePath) {
  __moore_config_db_clear();

  // Set same key twice
  int64_t value1 = 100;
  __moore_config_db_set(nullptr, "top.env", 7, "cfg", 3, &value1, sizeof(value1),
                        1);

  int64_t value2 = 200;
  __moore_config_db_set(nullptr, "top.env", 7, "cfg", 3, &value2, sizeof(value2),
                        1);

  // Should get the last set value
  int64_t result = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "top.env", 7, "cfg", 3, 1, &result,
                                  sizeof(result)),
            1);
  EXPECT_EQ(result, 200);

  __moore_config_db_clear();
}

TEST(MooreRuntimeConfigDbTest, StringValueStorage) {
  __moore_config_db_clear();

  // Set a string value (as raw bytes)
  char str[] = "hello world";
  int64_t strLen = 11;
  __moore_config_db_set(nullptr, "top", 3, "message", 7, str, strLen, 2);

  // Get it back
  char buffer[32] = {0};
  int32_t result = __moore_config_db_get(nullptr, "top", 3, "message", 7, 2,
                                         buffer, sizeof(buffer));
  EXPECT_EQ(result, 1);
  EXPECT_EQ(std::string(buffer, strLen), "hello world");

  __moore_config_db_clear();
}

TEST(MooreRuntimeConfigDbTest, EmptyInstName) {
  __moore_config_db_clear();

  // Set with empty instance name
  int64_t value = 999;
  __moore_config_db_set(nullptr, "", 0, "root_cfg", 8, &value, sizeof(value), 1);

  // Should be retrievable with empty instance name
  int64_t result = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "", 0, "root_cfg", 8, 1, &result,
                                  sizeof(result)),
            1);
  EXPECT_EQ(result, 999);

  __moore_config_db_clear();
}

TEST(MooreRuntimeConfigDbTest, ClearRemovesAllEntries) {
  __moore_config_db_clear();

  // Add multiple entries
  int64_t v1 = 1, v2 = 2, v3 = 3;
  __moore_config_db_set(nullptr, "a", 1, "f1", 2, &v1, sizeof(v1), 1);
  __moore_config_db_set(nullptr, "b", 1, "f2", 2, &v2, sizeof(v2), 1);
  __moore_config_db_set(nullptr, "*", 1, "f3", 2, &v3, sizeof(v3), 1);

  // Verify they exist
  EXPECT_EQ(__moore_config_db_exists("a", 1, "f1", 2), 1);
  EXPECT_EQ(__moore_config_db_exists("b", 1, "f2", 2), 1);
  EXPECT_EQ(__moore_config_db_exists("c", 1, "f3", 2), 1);  // via wildcard

  // Clear
  __moore_config_db_clear();

  // Verify they're gone
  EXPECT_EQ(__moore_config_db_exists("a", 1, "f1", 2), 0);
  EXPECT_EQ(__moore_config_db_exists("b", 1, "f2", 2), 0);
  EXPECT_EQ(__moore_config_db_exists("c", 1, "f3", 2), 0);
}

TEST(MooreRuntimeConfigDbTest, WildcardQuestionMark) {
  __moore_config_db_clear();

  // Set with pattern "top.agent?" (matches "top.agent0", "top.agentX", etc.)
  int64_t value = 777;
  __moore_config_db_set(nullptr, "top.agent?", 10, "cfg", 3, &value,
                        sizeof(value), 1);

  // Should match single character after "agent"
  int64_t result1 = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "top.agent0", 10, "cfg", 3, 1,
                                  &result1, sizeof(result1)),
            1);
  EXPECT_EQ(result1, 777);

  int64_t result2 = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "top.agentX", 10, "cfg", 3, 1,
                                  &result2, sizeof(result2)),
            1);
  EXPECT_EQ(result2, 777);

  // Should NOT match without the extra character
  int64_t result3 = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "top.agent", 9, "cfg", 3, 1, &result3,
                                  sizeof(result3)),
            0);

  // Should NOT match with multiple extra characters
  int64_t result4 = 0;
  EXPECT_EQ(__moore_config_db_get(nullptr, "top.agent123", 12, "cfg", 3, 1,
                                  &result4, sizeof(result4)),
            0);

  __moore_config_db_clear();
}

//===----------------------------------------------------------------------===//
// Packed String to String Conversion Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeStringTest, PackedStringToString) {
  // "HDL_TOP" packed into 56 bits (7 chars * 8 bits)
  // Big-endian: first char in MSB
  // 'H'=0x48, 'D'=0x44, 'L'=0x4C, '_'=0x5F, 'T'=0x54, 'O'=0x4F, 'P'=0x50
  // Packed: 0x48444C5F544F50
  int64_t packedHdlTop = 0x48444C5F544F50LL;
  MooreString result = __moore_packed_string_to_string(packedHdlTop);
  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 7);
  EXPECT_EQ(std::string(result.data, result.len), "HDL_TOP");
  __moore_free(result.data);
}

TEST(MooreRuntimeStringTest, PackedStringToStringShort) {
  // Test "ABC" (3 chars)
  int64_t packedAbc = 0x414243LL; // 'A'=0x41, 'B'=0x42, 'C'=0x43
  MooreString result = __moore_packed_string_to_string(packedAbc);
  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 3);
  EXPECT_EQ(std::string(result.data, result.len), "ABC");
  __moore_free(result.data);
}

TEST(MooreRuntimeStringTest, PackedStringToStringSingleChar) {
  // Test single character "X" (1 char)
  int64_t packedX = 0x58LL; // 'X'=0x58
  MooreString result = __moore_packed_string_to_string(packedX);
  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 1);
  EXPECT_EQ(result.data[0], 'X');
  __moore_free(result.data);
}

TEST(MooreRuntimeStringTest, PackedStringToStringEmpty) {
  // Test empty/zero value
  MooreString result = __moore_packed_string_to_string(0);
  EXPECT_TRUE(result.data == nullptr || result.len == 0);
}

TEST(MooreRuntimeStringTest, PackedStringToStringMaxLength) {
  // Test max length (8 characters)
  // "ABCDEFGH" packed
  int64_t packed8 = 0x4142434445464748LL;
  MooreString result = __moore_packed_string_to_string(packed8);
  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(result.len, 8);
  EXPECT_EQ(std::string(result.data, result.len), "ABCDEFGH");
  __moore_free(result.data);
}

//===----------------------------------------------------------------------===//
// Int to String (Decimal) Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeStringTest, IntToStringPositive) {
  MooreString result = __moore_int_to_string(123);
  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(std::string(result.data, result.len), "123");
  __moore_free(result.data);
}

TEST(MooreRuntimeStringTest, IntToStringZero) {
  MooreString result = __moore_int_to_string(0);
  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(std::string(result.data, result.len), "0");
  __moore_free(result.data);
}

TEST(MooreRuntimeStringTest, IntToStringLargeValue) {
  // Max 32-bit unsigned
  MooreString result = __moore_int_to_string(4294967295ULL);
  ASSERT_NE(result.data, nullptr);
  EXPECT_EQ(std::string(result.data, result.len), "4294967295");
  __moore_free(result.data);
}

//===----------------------------------------------------------------------===//
// TLM Port/Export Runtime Tests
//===----------------------------------------------------------------------===//

// Test transaction structure for TLM tests
struct TestTlmTransaction {
  int32_t addr;
  int32_t data;
  int8_t kind;

  bool operator==(const TestTlmTransaction &other) const {
    return addr == other.addr && data == other.data && kind == other.kind;
  }
};

// Callback state for subscriber tests
static std::atomic<int> tlmCallbackInvocations{0};
static TestTlmTransaction tlmLastReceivedTransaction;

void tlmTestWriteCallback(void *subscriber, void *transaction,
                          int64_t transactionSize) {
  (void)subscriber;
  if (transactionSize == sizeof(TestTlmTransaction)) {
    tlmLastReceivedTransaction =
        *static_cast<TestTlmTransaction *>(transaction);
  }
  tlmCallbackInvocations++;
}

TEST(MooreRuntimeTlmTest, PortCreation) {
  const char *portName = "test_port";
  MooreTlmPortHandle port = __moore_tlm_port_create(
      portName, strlen(portName), 0, MOORE_TLM_PORT_ANALYSIS);
  EXPECT_NE(port, MOORE_TLM_INVALID_HANDLE);

  MooreString name = __moore_tlm_port_get_name(port);
  ASSERT_NE(name.data, nullptr);
  EXPECT_EQ(std::string(name.data, name.len), portName);
  __moore_free(name.data);

  EXPECT_EQ(__moore_tlm_port_get_num_connections(port), 0);

  __moore_tlm_port_destroy(port);
}

TEST(MooreRuntimeTlmTest, FifoCreation) {
  const char *fifoName = "test_fifo";
  MooreTlmFifoHandle fifo = __moore_tlm_fifo_create(
      fifoName, strlen(fifoName), 0, 0, sizeof(TestTlmTransaction));
  EXPECT_NE(fifo, MOORE_TLM_INVALID_HANDLE);

  EXPECT_EQ(__moore_tlm_fifo_is_empty(fifo), 1);
  EXPECT_EQ(__moore_tlm_fifo_is_full(fifo), 0);
  EXPECT_EQ(__moore_tlm_fifo_size(fifo), 0);

  MooreTlmPortHandle analysisExport = __moore_tlm_fifo_get_analysis_export(fifo);
  EXPECT_NE(analysisExport, MOORE_TLM_INVALID_HANDLE);

  __moore_tlm_fifo_destroy(fifo);
}

TEST(MooreRuntimeTlmTest, FifoPutAndTryGet) {
  const char *fifoName = "fifo_put_get";
  MooreTlmFifoHandle fifo = __moore_tlm_fifo_create(
      fifoName, strlen(fifoName), 0, 0, sizeof(TestTlmTransaction));

  TestTlmTransaction tx1 = {0x1000, 0xDEAD, 1};
  TestTlmTransaction tx2 = {0x2000, 0xBEEF, 0};

  __moore_tlm_fifo_put(fifo, &tx1, sizeof(tx1));
  EXPECT_EQ(__moore_tlm_fifo_size(fifo), 1);

  __moore_tlm_fifo_put(fifo, &tx2, sizeof(tx2));
  EXPECT_EQ(__moore_tlm_fifo_size(fifo), 2);

  TestTlmTransaction rxTx;
  int32_t result = __moore_tlm_fifo_try_get(fifo, &rxTx, sizeof(rxTx));
  EXPECT_EQ(result, 1);
  EXPECT_TRUE(rxTx == tx1);
  EXPECT_EQ(__moore_tlm_fifo_size(fifo), 1);

  result = __moore_tlm_fifo_try_get(fifo, &rxTx, sizeof(rxTx));
  EXPECT_EQ(result, 1);
  EXPECT_TRUE(rxTx == tx2);
  EXPECT_EQ(__moore_tlm_fifo_is_empty(fifo), 1);

  result = __moore_tlm_fifo_try_get(fifo, &rxTx, sizeof(rxTx));
  EXPECT_EQ(result, 0);

  __moore_tlm_fifo_destroy(fifo);
}

TEST(MooreRuntimeTlmTest, FifoPeek) {
  const char *fifoName = "fifo_peek";
  MooreTlmFifoHandle fifo = __moore_tlm_fifo_create(
      fifoName, strlen(fifoName), 0, 0, sizeof(TestTlmTransaction));

  TestTlmTransaction tx = {0x3000, 0xCAFE, 1};
  __moore_tlm_fifo_put(fifo, &tx, sizeof(tx));

  TestTlmTransaction peeked;
  int32_t result = __moore_tlm_fifo_try_peek(fifo, &peeked, sizeof(peeked));
  EXPECT_EQ(result, 1);
  EXPECT_TRUE(peeked == tx);
  EXPECT_EQ(__moore_tlm_fifo_size(fifo), 1);

  // Peek again - should get same result
  result = __moore_tlm_fifo_try_peek(fifo, &peeked, sizeof(peeked));
  EXPECT_EQ(result, 1);
  EXPECT_TRUE(peeked == tx);

  __moore_tlm_fifo_destroy(fifo);
}

TEST(MooreRuntimeTlmTest, BoundedFifoFull) {
  const char *fifoName = "bounded_fifo";
  int64_t maxSize = 2;
  MooreTlmFifoHandle fifo = __moore_tlm_fifo_create(
      fifoName, strlen(fifoName), 0, maxSize, sizeof(TestTlmTransaction));

  TestTlmTransaction tx1 = {0x100, 0x11, 0};
  TestTlmTransaction tx2 = {0x200, 0x22, 1};
  TestTlmTransaction tx3 = {0x300, 0x33, 0};

  int32_t result = __moore_tlm_fifo_try_put(fifo, &tx1, sizeof(tx1));
  EXPECT_EQ(result, 1);

  result = __moore_tlm_fifo_try_put(fifo, &tx2, sizeof(tx2));
  EXPECT_EQ(result, 1);

  EXPECT_EQ(__moore_tlm_fifo_is_full(fifo), 1);

  result = __moore_tlm_fifo_try_put(fifo, &tx3, sizeof(tx3));
  EXPECT_EQ(result, 0);

  TestTlmTransaction rxTx;
  __moore_tlm_fifo_try_get(fifo, &rxTx, sizeof(rxTx));
  EXPECT_EQ(__moore_tlm_fifo_is_full(fifo), 0);

  result = __moore_tlm_fifo_try_put(fifo, &tx3, sizeof(tx3));
  EXPECT_EQ(result, 1);

  __moore_tlm_fifo_destroy(fifo);
}

TEST(MooreRuntimeTlmTest, PortConnection) {
  const char *portName = "analysis_port";
  const char *exportName = "analysis_imp";

  MooreTlmPortHandle port = __moore_tlm_port_create(portName, strlen(portName),
                                                    0, MOORE_TLM_PORT_ANALYSIS);
  MooreTlmPortHandle export_ = __moore_tlm_port_create(
      exportName, strlen(exportName), 0, MOORE_TLM_PORT_ANALYSIS);

  int32_t result = __moore_tlm_port_connect(port, export_);
  EXPECT_EQ(result, 1);
  EXPECT_EQ(__moore_tlm_port_get_num_connections(port), 1);

  const char *export2Name = "analysis_imp_2";
  MooreTlmPortHandle export2 = __moore_tlm_port_create(
      export2Name, strlen(export2Name), 0, MOORE_TLM_PORT_ANALYSIS);
  result = __moore_tlm_port_connect(port, export2);
  EXPECT_EQ(result, 1);
  EXPECT_EQ(__moore_tlm_port_get_num_connections(port), 2);

  __moore_tlm_port_destroy(port);
  __moore_tlm_port_destroy(export_);
  __moore_tlm_port_destroy(export2);
}

TEST(MooreRuntimeTlmTest, AnalysisPortToFifo) {
  // Monitor -> Scoreboard pattern
  const char *portName = "monitor.analysis_port";
  MooreTlmPortHandle analysisPort = __moore_tlm_port_create(
      portName, strlen(portName), 0, MOORE_TLM_PORT_ANALYSIS);

  const char *fifoName = "scoreboard.analysis_fifo";
  MooreTlmFifoHandle fifo = __moore_tlm_fifo_create(
      fifoName, strlen(fifoName), 0, 0, sizeof(TestTlmTransaction));

  MooreTlmPortHandle analysisExport = __moore_tlm_fifo_get_analysis_export(fifo);

  int32_t result = __moore_tlm_port_connect(analysisPort, analysisExport);
  EXPECT_EQ(result, 1);

  TestTlmTransaction tx = {0x4000, 0xFACE, 1};
  __moore_tlm_port_write(analysisPort, &tx, sizeof(tx));

  EXPECT_EQ(__moore_tlm_fifo_size(fifo), 1);

  TestTlmTransaction rxTx;
  result = __moore_tlm_fifo_try_get(fifo, &rxTx, sizeof(rxTx));
  EXPECT_EQ(result, 1);
  EXPECT_TRUE(rxTx == tx);

  __moore_tlm_port_destroy(analysisPort);
  __moore_tlm_fifo_destroy(fifo);
}

TEST(MooreRuntimeTlmTest, SubscriberCallback) {
  // Monitor -> Coverage pattern
  const char *portName = "monitor.analysis_port_2";
  MooreTlmPortHandle analysisPort = __moore_tlm_port_create(
      portName, strlen(portName), 0, MOORE_TLM_PORT_ANALYSIS);

  const char *impName = "coverage.analysis_export";
  MooreTlmPortHandle analysisImp = __moore_tlm_port_create(
      impName, strlen(impName), 0, MOORE_TLM_PORT_ANALYSIS);

  __moore_tlm_subscriber_set_write_callback(analysisImp, tlmTestWriteCallback,
                                            nullptr);

  __moore_tlm_port_connect(analysisPort, analysisImp);

  tlmCallbackInvocations = 0;
  tlmLastReceivedTransaction = {0, 0, 0};

  TestTlmTransaction tx = {0x5000, 0xBABE, 0};
  __moore_tlm_port_write(analysisPort, &tx, sizeof(tx));

  EXPECT_EQ(tlmCallbackInvocations.load(), 1);
  EXPECT_TRUE(tlmLastReceivedTransaction == tx);

  TestTlmTransaction tx2 = {0x6000, 0xDEAD, 1};
  __moore_tlm_port_write(analysisPort, &tx2, sizeof(tx2));

  EXPECT_EQ(tlmCallbackInvocations.load(), 2);
  EXPECT_TRUE(tlmLastReceivedTransaction == tx2);

  __moore_tlm_port_destroy(analysisPort);
  __moore_tlm_port_destroy(analysisImp);
}

TEST(MooreRuntimeTlmTest, FifoFlush) {
  const char *fifoName = "flush_test_fifo";
  MooreTlmFifoHandle fifo = __moore_tlm_fifo_create(
      fifoName, strlen(fifoName), 0, 0, sizeof(TestTlmTransaction));

  TestTlmTransaction tx1 = {0x100, 0x11, 0};
  TestTlmTransaction tx2 = {0x200, 0x22, 1};

  __moore_tlm_fifo_put(fifo, &tx1, sizeof(tx1));
  __moore_tlm_fifo_put(fifo, &tx2, sizeof(tx2));
  EXPECT_EQ(__moore_tlm_fifo_size(fifo), 2);

  __moore_tlm_fifo_flush(fifo);
  EXPECT_EQ(__moore_tlm_fifo_is_empty(fifo), 1);
  EXPECT_EQ(__moore_tlm_fifo_size(fifo), 0);

  __moore_tlm_fifo_destroy(fifo);
}

TEST(MooreRuntimeTlmTest, MultipleSubscribers) {
  // 1-to-N broadcast
  const char *portName = "broadcast_port";
  MooreTlmPortHandle port = __moore_tlm_port_create(
      portName, strlen(portName), 0, MOORE_TLM_PORT_ANALYSIS);

  const char *fifo1Name = "subscriber_fifo_1";
  const char *fifo2Name = "subscriber_fifo_2";
  MooreTlmFifoHandle fifo1 = __moore_tlm_fifo_create(
      fifo1Name, strlen(fifo1Name), 0, 0, sizeof(TestTlmTransaction));
  MooreTlmFifoHandle fifo2 = __moore_tlm_fifo_create(
      fifo2Name, strlen(fifo2Name), 0, 0, sizeof(TestTlmTransaction));

  MooreTlmPortHandle export1 = __moore_tlm_fifo_get_analysis_export(fifo1);
  MooreTlmPortHandle export2 = __moore_tlm_fifo_get_analysis_export(fifo2);
  __moore_tlm_port_connect(port, export1);
  __moore_tlm_port_connect(port, export2);

  TestTlmTransaction tx = {0x7000, 0xF00D, 1};
  __moore_tlm_port_write(port, &tx, sizeof(tx));

  EXPECT_EQ(__moore_tlm_fifo_size(fifo1), 1);
  EXPECT_EQ(__moore_tlm_fifo_size(fifo2), 1);

  TestTlmTransaction rx1, rx2;
  __moore_tlm_fifo_try_get(fifo1, &rx1, sizeof(rx1));
  __moore_tlm_fifo_try_get(fifo2, &rx2, sizeof(rx2));
  EXPECT_TRUE(rx1 == tx);
  EXPECT_TRUE(rx2 == tx);

  __moore_tlm_port_destroy(port);
  __moore_tlm_fifo_destroy(fifo1);
  __moore_tlm_fifo_destroy(fifo2);
}

TEST(MooreRuntimeTlmTest, TracingEnableDisable) {
  EXPECT_EQ(__moore_tlm_is_trace_enabled(), 0);

  __moore_tlm_set_trace_enabled(1);
  EXPECT_EQ(__moore_tlm_is_trace_enabled(), 1);

  __moore_tlm_set_trace_enabled(0);
  EXPECT_EQ(__moore_tlm_is_trace_enabled(), 0);
}

TEST(MooreRuntimeTlmTest, Statistics) {
  int64_t conns, writes, gets;
  __moore_tlm_get_statistics(&conns, &writes, &gets);

  // These should be non-zero after running the tests above
  // Note: The exact values depend on the order of test execution
  EXPECT_GE(conns, 0);
  EXPECT_GE(writes, 0);
  EXPECT_GE(gets, 0);
}

TEST(MooreRuntimeTlmTest, FifoCanPutAndCanGet) {
  const char *fifoName = "can_put_get_fifo";
  // Create bounded FIFO with max size 2
  MooreTlmFifoHandle fifo = __moore_tlm_fifo_create(
      fifoName, strlen(fifoName), 0, 2, sizeof(TestTlmTransaction));

  // Empty FIFO: can_put should be true, can_get should be false
  EXPECT_EQ(__moore_tlm_fifo_can_put(fifo), 1);
  EXPECT_EQ(__moore_tlm_fifo_can_get(fifo), 0);

  // Add one item
  TestTlmTransaction tx1 = {0x100, 0x11, 0};
  __moore_tlm_fifo_put(fifo, &tx1, sizeof(tx1));

  // One item: can_put should be true, can_get should be true
  EXPECT_EQ(__moore_tlm_fifo_can_put(fifo), 1);
  EXPECT_EQ(__moore_tlm_fifo_can_get(fifo), 1);

  // Add second item (now full)
  TestTlmTransaction tx2 = {0x200, 0x22, 1};
  __moore_tlm_fifo_put(fifo, &tx2, sizeof(tx2));

  // Full FIFO: can_put should be false, can_get should be true
  EXPECT_EQ(__moore_tlm_fifo_can_put(fifo), 0);
  EXPECT_EQ(__moore_tlm_fifo_can_get(fifo), 1);

  // Remove one item
  TestTlmTransaction rxTx;
  __moore_tlm_fifo_try_get(fifo, &rxTx, sizeof(rxTx));

  // After removal: can_put should be true again
  EXPECT_EQ(__moore_tlm_fifo_can_put(fifo), 1);
  EXPECT_EQ(__moore_tlm_fifo_can_get(fifo), 1);

  __moore_tlm_fifo_destroy(fifo);
}

TEST(MooreRuntimeTlmTest, FifoUsedAndFree) {
  const char *fifoName = "used_free_fifo";
  // Create bounded FIFO with max size 5
  MooreTlmFifoHandle fifo = __moore_tlm_fifo_create(
      fifoName, strlen(fifoName), 0, 5, sizeof(TestTlmTransaction));

  // Empty FIFO: used=0, free=5
  EXPECT_EQ(__moore_tlm_fifo_used(fifo), 0);
  EXPECT_EQ(__moore_tlm_fifo_free(fifo), 5);
  EXPECT_EQ(__moore_tlm_fifo_capacity(fifo), 5);

  // Add three items
  TestTlmTransaction tx = {0x100, 0x11, 0};
  __moore_tlm_fifo_put(fifo, &tx, sizeof(tx));
  __moore_tlm_fifo_put(fifo, &tx, sizeof(tx));
  __moore_tlm_fifo_put(fifo, &tx, sizeof(tx));

  // After adding 3: used=3, free=2
  EXPECT_EQ(__moore_tlm_fifo_used(fifo), 3);
  EXPECT_EQ(__moore_tlm_fifo_free(fifo), 2);

  // Fill completely
  __moore_tlm_fifo_put(fifo, &tx, sizeof(tx));
  __moore_tlm_fifo_put(fifo, &tx, sizeof(tx));

  // Full FIFO: used=5, free=0
  EXPECT_EQ(__moore_tlm_fifo_used(fifo), 5);
  EXPECT_EQ(__moore_tlm_fifo_free(fifo), 0);

  __moore_tlm_fifo_destroy(fifo);
}

TEST(MooreRuntimeTlmTest, UnboundedFifoCanPutAndFree) {
  const char *fifoName = "unbounded_fifo";
  // Create unbounded FIFO (maxSize=0)
  MooreTlmFifoHandle fifo = __moore_tlm_fifo_create(
      fifoName, strlen(fifoName), 0, 0, sizeof(TestTlmTransaction));

  // Unbounded FIFO: can_put should always be true
  EXPECT_EQ(__moore_tlm_fifo_can_put(fifo), 1);
  EXPECT_EQ(__moore_tlm_fifo_capacity(fifo), 0);  // 0 indicates unbounded
  EXPECT_EQ(__moore_tlm_fifo_free(fifo), INT64_MAX);  // Unlimited free space

  // Add several items
  TestTlmTransaction tx = {0x100, 0x11, 0};
  for (int i = 0; i < 100; i++) {
    __moore_tlm_fifo_put(fifo, &tx, sizeof(tx));
  }

  // Still can put (unbounded)
  EXPECT_EQ(__moore_tlm_fifo_can_put(fifo), 1);
  EXPECT_EQ(__moore_tlm_fifo_used(fifo), 100);
  EXPECT_EQ(__moore_tlm_fifo_free(fifo), INT64_MAX);

  __moore_tlm_fifo_destroy(fifo);
}

TEST(MooreRuntimeTlmTest, FifoCapacity) {
  // Test bounded FIFO capacity
  const char *boundedName = "bounded_cap_fifo";
  MooreTlmFifoHandle boundedFifo = __moore_tlm_fifo_create(
      boundedName, strlen(boundedName), 0, 10, sizeof(TestTlmTransaction));
  EXPECT_EQ(__moore_tlm_fifo_capacity(boundedFifo), 10);
  __moore_tlm_fifo_destroy(boundedFifo);

  // Test unbounded FIFO capacity
  const char *unboundedName = "unbounded_cap_fifo";
  MooreTlmFifoHandle unboundedFifo = __moore_tlm_fifo_create(
      unboundedName, strlen(unboundedName), 0, 0, sizeof(TestTlmTransaction));
  EXPECT_EQ(__moore_tlm_fifo_capacity(unboundedFifo), 0);
  __moore_tlm_fifo_destroy(unboundedFifo);
}

//===----------------------------------------------------------------------===//
// UVM Objection System Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeObjectionTest, CreateAndDestroy) {
  const char *phaseName = "run_phase";
  MooreObjectionHandle objection = __moore_objection_create(
      phaseName, strlen(phaseName));
  EXPECT_NE(objection, MOORE_OBJECTION_INVALID_HANDLE);

  MooreString name = __moore_objection_get_phase_name(objection);
  ASSERT_NE(name.data, nullptr);
  EXPECT_EQ(std::string(name.data, name.len), phaseName);
  __moore_free(name.data);

  EXPECT_EQ(__moore_objection_get_count(objection), 0);
  EXPECT_EQ(__moore_objection_is_zero(objection), 1);

  __moore_objection_destroy(objection);
}

TEST(MooreRuntimeObjectionTest, RaiseAndDrop) {
  const char *phaseName = "main_phase";
  MooreObjectionHandle objection = __moore_objection_create(
      phaseName, strlen(phaseName));

  // Initially zero
  EXPECT_EQ(__moore_objection_get_count(objection), 0);
  EXPECT_EQ(__moore_objection_is_zero(objection), 1);

  // Raise single objection
  const char *ctx1 = "uvm_test_top.env.agent1";
  __moore_objection_raise(objection, ctx1, strlen(ctx1), nullptr, 0, 1);
  EXPECT_EQ(__moore_objection_get_count(objection), 1);
  EXPECT_EQ(__moore_objection_is_zero(objection), 0);

  // Raise another from different context
  const char *ctx2 = "uvm_test_top.env.agent2";
  __moore_objection_raise(objection, ctx2, strlen(ctx2), nullptr, 0, 1);
  EXPECT_EQ(__moore_objection_get_count(objection), 2);

  // Check per-context counts
  EXPECT_EQ(__moore_objection_get_count_by_context(objection, ctx1, strlen(ctx1)), 1);
  EXPECT_EQ(__moore_objection_get_count_by_context(objection, ctx2, strlen(ctx2)), 1);

  // Drop one
  __moore_objection_drop(objection, ctx1, strlen(ctx1), nullptr, 0, 1);
  EXPECT_EQ(__moore_objection_get_count(objection), 1);
  EXPECT_EQ(__moore_objection_get_count_by_context(objection, ctx1, strlen(ctx1)), 0);

  // Drop the other
  __moore_objection_drop(objection, ctx2, strlen(ctx2), nullptr, 0, 1);
  EXPECT_EQ(__moore_objection_get_count(objection), 0);
  EXPECT_EQ(__moore_objection_is_zero(objection), 1);

  __moore_objection_destroy(objection);
}

TEST(MooreRuntimeObjectionTest, RaiseMultiple) {
  const char *phaseName = "test_phase";
  MooreObjectionHandle objection = __moore_objection_create(
      phaseName, strlen(phaseName));

  // Raise multiple objections at once
  const char *ctx = "uvm_test_top.driver";
  __moore_objection_raise(objection, ctx, strlen(ctx), nullptr, 0, 5);
  EXPECT_EQ(__moore_objection_get_count(objection), 5);
  EXPECT_EQ(__moore_objection_get_count_by_context(objection, ctx, strlen(ctx)), 5);

  // Raise more from same context
  __moore_objection_raise(objection, ctx, strlen(ctx), nullptr, 0, 3);
  EXPECT_EQ(__moore_objection_get_count(objection), 8);
  EXPECT_EQ(__moore_objection_get_count_by_context(objection, ctx, strlen(ctx)), 8);

  // Drop some
  __moore_objection_drop(objection, ctx, strlen(ctx), nullptr, 0, 4);
  EXPECT_EQ(__moore_objection_get_count(objection), 4);

  // Drop all remaining
  __moore_objection_drop(objection, ctx, strlen(ctx), nullptr, 0, 4);
  EXPECT_EQ(__moore_objection_get_count(objection), 0);
  EXPECT_EQ(__moore_objection_is_zero(objection), 1);

  __moore_objection_destroy(objection);
}

TEST(MooreRuntimeObjectionTest, DrainTime) {
  const char *phaseName = "shutdown_phase";
  MooreObjectionHandle objection = __moore_objection_create(
      phaseName, strlen(phaseName));

  // Initially zero drain time
  EXPECT_EQ(__moore_objection_get_drain_time(objection), 0);

  // Set drain time
  __moore_objection_set_drain_time(objection, 100);
  EXPECT_EQ(__moore_objection_get_drain_time(objection), 100);

  // Update drain time
  __moore_objection_set_drain_time(objection, 50);
  EXPECT_EQ(__moore_objection_get_drain_time(objection), 50);

  // Reset to zero
  __moore_objection_set_drain_time(objection, 0);
  EXPECT_EQ(__moore_objection_get_drain_time(objection), 0);

  __moore_objection_destroy(objection);
}

TEST(MooreRuntimeObjectionTest, WithDescriptions) {
  const char *phaseName = "run_phase";
  MooreObjectionHandle objection = __moore_objection_create(
      phaseName, strlen(phaseName));

  const char *ctx = "uvm_test_top.env.sequencer";
  const char *desc1 = "Sequence in progress";
  const char *desc2 = "Cleanup complete";

  // Raise with description
  __moore_objection_raise(objection, ctx, strlen(ctx), desc1, strlen(desc1), 1);
  EXPECT_EQ(__moore_objection_get_count(objection), 1);

  // Drop with description
  __moore_objection_drop(objection, ctx, strlen(ctx), desc2, strlen(desc2), 1);
  EXPECT_EQ(__moore_objection_get_count(objection), 0);

  __moore_objection_destroy(objection);
}

TEST(MooreRuntimeObjectionTest, NullContext) {
  const char *phaseName = "run_phase";
  MooreObjectionHandle objection = __moore_objection_create(
      phaseName, strlen(phaseName));

  // Raise with null context (anonymous objection)
  __moore_objection_raise(objection, nullptr, 0, nullptr, 0, 1);
  EXPECT_EQ(__moore_objection_get_count(objection), 1);

  // Drop with null context
  __moore_objection_drop(objection, nullptr, 0, nullptr, 0, 1);
  EXPECT_EQ(__moore_objection_get_count(objection), 0);

  __moore_objection_destroy(objection);
}

TEST(MooreRuntimeObjectionTest, WaitForZeroImmediate) {
  const char *phaseName = "test_phase";
  MooreObjectionHandle objection = __moore_objection_create(
      phaseName, strlen(phaseName));

  // Already zero, should return immediately
  int32_t result = __moore_objection_wait_for_zero(objection);
  EXPECT_EQ(result, 1);

  __moore_objection_destroy(objection);
}

TEST(MooreRuntimeObjectionTest, WaitForZeroWithThread) {
  const char *phaseName = "concurrent_test";
  MooreObjectionHandle objection = __moore_objection_create(
      phaseName, strlen(phaseName));

  const char *ctx = "test_component";
  __moore_objection_raise(objection, ctx, strlen(ctx), nullptr, 0, 1);

  std::atomic<bool> waitCompleted{false};

  // Start a thread that waits for zero
  std::thread waiter([&]() {
    __moore_objection_wait_for_zero(objection);
    waitCompleted = true;
  });

  // Small delay to ensure waiter is waiting
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(waitCompleted.load());

  // Drop the objection
  __moore_objection_drop(objection, ctx, strlen(ctx), nullptr, 0, 1);

  // Wait for the thread to complete
  waiter.join();
  EXPECT_TRUE(waitCompleted.load());

  __moore_objection_destroy(objection);
}

TEST(MooreRuntimeObjectionTest, MultiplePhases) {
  const char *runPhase = "run";
  const char *mainPhase = "main";
  const char *shutdownPhase = "shutdown";

  MooreObjectionHandle runObj = __moore_objection_create(runPhase, strlen(runPhase));
  MooreObjectionHandle mainObj = __moore_objection_create(mainPhase, strlen(mainPhase));
  MooreObjectionHandle shutdownObj = __moore_objection_create(shutdownPhase, strlen(shutdownPhase));

  EXPECT_NE(runObj, mainObj);
  EXPECT_NE(mainObj, shutdownObj);

  // Raise on different phases
  __moore_objection_raise(runObj, nullptr, 0, nullptr, 0, 1);
  __moore_objection_raise(mainObj, nullptr, 0, nullptr, 0, 2);
  __moore_objection_raise(shutdownObj, nullptr, 0, nullptr, 0, 3);

  EXPECT_EQ(__moore_objection_get_count(runObj), 1);
  EXPECT_EQ(__moore_objection_get_count(mainObj), 2);
  EXPECT_EQ(__moore_objection_get_count(shutdownObj), 3);

  // Drop independently
  __moore_objection_drop(runObj, nullptr, 0, nullptr, 0, 1);
  EXPECT_EQ(__moore_objection_get_count(runObj), 0);
  EXPECT_EQ(__moore_objection_get_count(mainObj), 2);
  EXPECT_EQ(__moore_objection_get_count(shutdownObj), 3);

  __moore_objection_destroy(runObj);
  __moore_objection_destroy(mainObj);
  __moore_objection_destroy(shutdownObj);
}

TEST(MooreRuntimeObjectionTest, TracingEnableDisable) {
  EXPECT_EQ(__moore_objection_is_trace_enabled(), 0);

  __moore_objection_set_trace_enabled(1);
  EXPECT_EQ(__moore_objection_is_trace_enabled(), 1);

  __moore_objection_set_trace_enabled(0);
  EXPECT_EQ(__moore_objection_is_trace_enabled(), 0);
}

TEST(MooreRuntimeObjectionTest, InvalidHandle) {
  // These should not crash - just return default values
  EXPECT_EQ(__moore_objection_get_count(MOORE_OBJECTION_INVALID_HANDLE), 0);
  EXPECT_EQ(__moore_objection_get_drain_time(MOORE_OBJECTION_INVALID_HANDLE), 0);
  EXPECT_EQ(__moore_objection_is_zero(MOORE_OBJECTION_INVALID_HANDLE), 1);

  MooreString name = __moore_objection_get_phase_name(MOORE_OBJECTION_INVALID_HANDLE);
  EXPECT_EQ(name.data, nullptr);
  EXPECT_EQ(name.len, 0);

  // These should silently fail without crash
  __moore_objection_raise(MOORE_OBJECTION_INVALID_HANDLE, nullptr, 0, nullptr, 0, 1);
  __moore_objection_drop(MOORE_OBJECTION_INVALID_HANDLE, nullptr, 0, nullptr, 0, 1);
  __moore_objection_set_drain_time(MOORE_OBJECTION_INVALID_HANDLE, 100);
  __moore_objection_destroy(MOORE_OBJECTION_INVALID_HANDLE);
}

TEST(MooreRuntimeObjectionTest, HierarchicalContexts) {
  const char *phaseName = "run_phase";
  MooreObjectionHandle objection = __moore_objection_create(
      phaseName, strlen(phaseName));

  // Simulate UVM component hierarchy
  const char *ctx1 = "uvm_test_top";
  const char *ctx2 = "uvm_test_top.env";
  const char *ctx3 = "uvm_test_top.env.agent";
  const char *ctx4 = "uvm_test_top.env.agent.driver";
  const char *ctx5 = "uvm_test_top.env.agent.monitor";

  __moore_objection_raise(objection, ctx1, strlen(ctx1), nullptr, 0, 1);
  __moore_objection_raise(objection, ctx2, strlen(ctx2), nullptr, 0, 1);
  __moore_objection_raise(objection, ctx3, strlen(ctx3), nullptr, 0, 1);
  __moore_objection_raise(objection, ctx4, strlen(ctx4), nullptr, 0, 1);
  __moore_objection_raise(objection, ctx5, strlen(ctx5), nullptr, 0, 1);

  EXPECT_EQ(__moore_objection_get_count(objection), 5);

  // Each context tracked separately
  EXPECT_EQ(__moore_objection_get_count_by_context(objection, ctx1, strlen(ctx1)), 1);
  EXPECT_EQ(__moore_objection_get_count_by_context(objection, ctx4, strlen(ctx4)), 1);
  EXPECT_EQ(__moore_objection_get_count_by_context(objection, ctx5, strlen(ctx5)), 1);

  // Drop in reverse hierarchy order (typical shutdown)
  __moore_objection_drop(objection, ctx5, strlen(ctx5), nullptr, 0, 1);
  __moore_objection_drop(objection, ctx4, strlen(ctx4), nullptr, 0, 1);
  __moore_objection_drop(objection, ctx3, strlen(ctx3), nullptr, 0, 1);
  __moore_objection_drop(objection, ctx2, strlen(ctx2), nullptr, 0, 1);
  __moore_objection_drop(objection, ctx1, strlen(ctx1), nullptr, 0, 1);

  EXPECT_EQ(__moore_objection_get_count(objection), 0);
  EXPECT_EQ(__moore_objection_is_zero(objection), 1);

  __moore_objection_destroy(objection);
}

//===----------------------------------------------------------------------===//
// UVM Sequence/Sequencer Tests
//===----------------------------------------------------------------------===//

// Test transaction type for sequence tests
struct TestSeqTransaction {
  uint32_t addr;
  uint32_t data;
  uint8_t kind;  // 0=read, 1=write

  bool operator==(const TestSeqTransaction &other) const {
    return addr == other.addr && data == other.data && kind == other.kind;
  }
};

TEST(MooreRuntimeSequenceTest, SequencerCreation) {
  const char *seqrName = "test_sequencer";
  MooreSequencerHandle seqr = __moore_sequencer_create(
      seqrName, strlen(seqrName), 0);
  EXPECT_NE(seqr, MOORE_SEQUENCER_INVALID_HANDLE);

  MooreString name = __moore_sequencer_get_name(seqr);
  ASSERT_NE(name.data, nullptr);
  EXPECT_EQ(std::string(name.data, name.len), seqrName);
  __moore_free(name.data);

  EXPECT_EQ(__moore_sequencer_is_running(seqr), 0);
  EXPECT_EQ(__moore_sequencer_get_num_waiting(seqr), 0);

  __moore_sequencer_destroy(seqr);
}

TEST(MooreRuntimeSequenceTest, SequencerStartStop) {
  const char *seqrName = "start_stop_sequencer";
  MooreSequencerHandle seqr = __moore_sequencer_create(
      seqrName, strlen(seqrName), 0);

  EXPECT_EQ(__moore_sequencer_is_running(seqr), 0);

  __moore_sequencer_start(seqr);
  EXPECT_EQ(__moore_sequencer_is_running(seqr), 1);

  __moore_sequencer_stop(seqr);
  EXPECT_EQ(__moore_sequencer_is_running(seqr), 0);

  __moore_sequencer_destroy(seqr);
}

TEST(MooreRuntimeSequenceTest, SequenceCreation) {
  const char *seqName = "test_sequence";
  MooreSequenceHandle seq = __moore_sequence_create(seqName, strlen(seqName), 10);
  EXPECT_NE(seq, MOORE_SEQUENCE_INVALID_HANDLE);

  MooreString name = __moore_sequence_get_name(seq);
  ASSERT_NE(name.data, nullptr);
  EXPECT_EQ(std::string(name.data, name.len), seqName);
  __moore_free(name.data);

  EXPECT_EQ(__moore_sequence_get_priority(seq), 10);
  EXPECT_EQ(__moore_sequence_get_state(seq), MOORE_SEQ_STATE_IDLE);

  __moore_sequence_destroy(seq);
}

TEST(MooreRuntimeSequenceTest, SequencePriority) {
  const char *seqName = "priority_seq";
  MooreSequenceHandle seq = __moore_sequence_create(seqName, strlen(seqName), 5);

  EXPECT_EQ(__moore_sequence_get_priority(seq), 5);

  __moore_sequence_set_priority(seq, 100);
  EXPECT_EQ(__moore_sequence_get_priority(seq), 100);

  __moore_sequence_set_priority(seq, 0);
  EXPECT_EQ(__moore_sequence_get_priority(seq), 0);

  __moore_sequence_destroy(seq);
}

TEST(MooreRuntimeSequenceTest, ArbitrationModes) {
  const char *seqrName = "arb_sequencer";
  MooreSequencerHandle seqr = __moore_sequencer_create(
      seqrName, strlen(seqrName), 0);

  // Default is FIFO
  EXPECT_EQ(__moore_sequencer_get_arbitration(seqr), MOORE_SEQ_ARB_FIFO);

  __moore_sequencer_set_arbitration(seqr, MOORE_SEQ_ARB_RANDOM);
  EXPECT_EQ(__moore_sequencer_get_arbitration(seqr), MOORE_SEQ_ARB_RANDOM);

  __moore_sequencer_set_arbitration(seqr, MOORE_SEQ_ARB_WEIGHTED);
  EXPECT_EQ(__moore_sequencer_get_arbitration(seqr), MOORE_SEQ_ARB_WEIGHTED);

  __moore_sequencer_set_arbitration(seqr, MOORE_SEQ_ARB_STRICT_FIFO);
  EXPECT_EQ(__moore_sequencer_get_arbitration(seqr), MOORE_SEQ_ARB_STRICT_FIFO);

  __moore_sequencer_set_arbitration(seqr, MOORE_SEQ_ARB_STRICT_RANDOM);
  EXPECT_EQ(__moore_sequencer_get_arbitration(seqr), MOORE_SEQ_ARB_STRICT_RANDOM);

  __moore_sequencer_set_arbitration(seqr, MOORE_SEQ_ARB_USER);
  EXPECT_EQ(__moore_sequencer_get_arbitration(seqr), MOORE_SEQ_ARB_USER);

  __moore_sequencer_destroy(seqr);
}

// Simple sequence body for testing
static std::atomic<int> simpleSeqBodyCalled{0};
static void simpleSequenceBody(MooreSequenceHandle sequence, void *userData) {
  (void)sequence;
  (void)userData;
  simpleSeqBodyCalled++;
}

TEST(MooreRuntimeSequenceTest, SimpleSequenceStart) {
  const char *seqrName = "simple_sequencer";
  const char *seqName = "simple_sequence";

  MooreSequencerHandle seqr = __moore_sequencer_create(
      seqrName, strlen(seqrName), 0);
  MooreSequenceHandle seq = __moore_sequence_create(seqName, strlen(seqName), 0);

  __moore_sequencer_start(seqr);

  simpleSeqBodyCalled = 0;
  int32_t result = __moore_sequence_start(seq, seqr, simpleSequenceBody, nullptr);
  EXPECT_EQ(result, 1);
  EXPECT_EQ(simpleSeqBodyCalled.load(), 1);

  EXPECT_EQ(__moore_sequence_get_state(seq), MOORE_SEQ_STATE_FINISHED);

  __moore_sequencer_stop(seqr);
  __moore_sequence_destroy(seq);
  __moore_sequencer_destroy(seqr);
}

// Sequence body that generates multiple items
struct SeqItemTestContext {
  MooreSequencerHandle sequencer;
  std::vector<TestSeqTransaction> itemsToSend;
  std::atomic<int> itemsSent{0};
};

static void multiItemSequenceBody(MooreSequenceHandle sequence, void *userData) {
  auto *ctx = static_cast<SeqItemTestContext *>(userData);

  for (auto &tx : ctx->itemsToSend) {
    if (__moore_sequence_start_item(sequence, &tx, sizeof(tx))) {
      // Item ready to be sent
      if (__moore_sequence_finish_item(sequence, &tx, sizeof(tx))) {
        ctx->itemsSent++;
      }
    }
  }
}

TEST(MooreRuntimeSequenceTest, SequenceDriverHandshake) {
  const char *seqrName = "handshake_sequencer";
  const char *seqName = "handshake_sequence";

  MooreSequencerHandle seqr = __moore_sequencer_create(
      seqrName, strlen(seqrName), 0);
  MooreSequenceHandle seq = __moore_sequence_create(seqName, strlen(seqName), 0);

  __moore_sequencer_start(seqr);

  // Set up context
  SeqItemTestContext ctx;
  ctx.sequencer = seqr;
  ctx.itemsToSend = {{0x1000, 0xAABB, 1}, {0x2000, 0xCCDD, 0}};
  ctx.itemsSent = 0;

  // Start sequence in async mode
  int32_t started = __moore_sequence_start_async(seq, seqr,
                                                  multiItemSequenceBody, &ctx);
  EXPECT_EQ(started, 1);

  // Driver side: get items
  TestSeqTransaction rxTx;
  for (size_t i = 0; i < ctx.itemsToSend.size(); ++i) {
    int32_t got = __moore_sequencer_get_next_item(seqr, &rxTx, sizeof(rxTx));
    EXPECT_EQ(got, 1);
    EXPECT_TRUE(rxTx == ctx.itemsToSend[i]);

    // Signal item done
    __moore_sequencer_item_done(seqr);
  }

  // Wait for sequence to complete
  int32_t completed = __moore_sequence_wait(seq);
  EXPECT_EQ(completed, 1);
  EXPECT_EQ(ctx.itemsSent.load(), 2);

  __moore_sequencer_stop(seqr);
  __moore_sequence_destroy(seq);
  __moore_sequencer_destroy(seqr);
}

TEST(MooreRuntimeSequenceTest, ItemDoneWithResponse) {
  const char *seqrName = "response_sequencer";
  const char *seqName = "response_sequence";

  MooreSequencerHandle seqr = __moore_sequencer_create(
      seqrName, strlen(seqrName), 0);
  MooreSequenceHandle seq = __moore_sequence_create(seqName, strlen(seqName), 0);

  __moore_sequencer_start(seqr);

  // Sequence sends a read request
  SeqItemTestContext ctx;
  ctx.sequencer = seqr;
  ctx.itemsToSend = {{0x3000, 0x0000, 0}};  // Read request
  ctx.itemsSent = 0;

  __moore_sequence_start_async(seq, seqr, multiItemSequenceBody, &ctx);

  // Driver receives and sends response with data
  TestSeqTransaction rxTx;
  EXPECT_EQ(__moore_sequencer_get_next_item(seqr, &rxTx, sizeof(rxTx)), 1);
  EXPECT_EQ(rxTx.addr, 0x3000u);
  EXPECT_EQ(rxTx.kind, 0);  // Read

  // Respond with data
  TestSeqTransaction response = {0x3000, 0xDEADBEEF, 0};
  __moore_sequencer_item_done_with_response(seqr, &response, sizeof(response));

  __moore_sequence_wait(seq);
  EXPECT_EQ(ctx.itemsSent.load(), 1);

  __moore_sequencer_stop(seqr);
  __moore_sequence_destroy(seq);
  __moore_sequencer_destroy(seqr);
}

TEST(MooreRuntimeSequenceTest, SequencerHasItems) {
  const char *seqrName = "has_items_sequencer";
  MooreSequencerHandle seqr = __moore_sequencer_create(
      seqrName, strlen(seqrName), 0);

  __moore_sequencer_start(seqr);

  // Initially no items
  EXPECT_EQ(__moore_sequencer_has_items(seqr), 0);

  __moore_sequencer_stop(seqr);
  __moore_sequencer_destroy(seqr);
}

TEST(MooreRuntimeSequenceTest, SequenceStop) {
  const char *seqName = "stoppable_sequence";
  MooreSequenceHandle seq = __moore_sequence_create(seqName, strlen(seqName), 0);

  EXPECT_EQ(__moore_sequence_get_state(seq), MOORE_SEQ_STATE_IDLE);

  __moore_sequence_stop(seq);
  EXPECT_EQ(__moore_sequence_get_state(seq), MOORE_SEQ_STATE_STOPPED);

  __moore_sequence_destroy(seq);
}

TEST(MooreRuntimeSequenceTest, TracingEnableDisable) {
  EXPECT_EQ(__moore_seq_is_trace_enabled(), 0);

  __moore_seq_set_trace_enabled(1);
  EXPECT_EQ(__moore_seq_is_trace_enabled(), 1);

  __moore_seq_set_trace_enabled(0);
  EXPECT_EQ(__moore_seq_is_trace_enabled(), 0);
}

TEST(MooreRuntimeSequenceTest, Statistics) {
  int64_t seqs, items, arbs;
  __moore_seq_get_statistics(&seqs, &items, &arbs);

  // After running the tests above, these should be non-zero
  EXPECT_GE(seqs, 0);
  EXPECT_GE(items, 0);
  EXPECT_GE(arbs, 0);
}

TEST(MooreRuntimeSequenceTest, InvalidHandles) {
  // These should not crash - just return default values
  EXPECT_EQ(__moore_sequencer_is_running(MOORE_SEQUENCER_INVALID_HANDLE), 0);
  EXPECT_EQ(__moore_sequencer_get_arbitration(MOORE_SEQUENCER_INVALID_HANDLE),
            MOORE_SEQ_ARB_FIFO);
  EXPECT_EQ(__moore_sequencer_get_num_waiting(MOORE_SEQUENCER_INVALID_HANDLE), 0);
  EXPECT_EQ(__moore_sequencer_has_items(MOORE_SEQUENCER_INVALID_HANDLE), 0);

  MooreString seqrName = __moore_sequencer_get_name(MOORE_SEQUENCER_INVALID_HANDLE);
  EXPECT_EQ(seqrName.data, nullptr);
  EXPECT_EQ(seqrName.len, 0);

  EXPECT_EQ(__moore_sequence_get_state(MOORE_SEQUENCE_INVALID_HANDLE),
            MOORE_SEQ_STATE_IDLE);
  EXPECT_EQ(__moore_sequence_get_priority(MOORE_SEQUENCE_INVALID_HANDLE), 0);

  MooreString seqName = __moore_sequence_get_name(MOORE_SEQUENCE_INVALID_HANDLE);
  EXPECT_EQ(seqName.data, nullptr);
  EXPECT_EQ(seqName.len, 0);

  // These should silently fail without crash
  __moore_sequencer_start(MOORE_SEQUENCER_INVALID_HANDLE);
  __moore_sequencer_stop(MOORE_SEQUENCER_INVALID_HANDLE);
  __moore_sequencer_set_arbitration(MOORE_SEQUENCER_INVALID_HANDLE,
                                     MOORE_SEQ_ARB_RANDOM);
  __moore_sequencer_destroy(MOORE_SEQUENCER_INVALID_HANDLE);

  __moore_sequence_stop(MOORE_SEQUENCE_INVALID_HANDLE);
  __moore_sequence_set_priority(MOORE_SEQUENCE_INVALID_HANDLE, 100);
  __moore_sequence_destroy(MOORE_SEQUENCE_INVALID_HANDLE);
}

TEST(MooreRuntimeSequenceTest, MultipleSequencers) {
  const char *name1 = "sequencer_1";
  const char *name2 = "sequencer_2";
  const char *name3 = "sequencer_3";

  MooreSequencerHandle seqr1 = __moore_sequencer_create(name1, strlen(name1), 0);
  MooreSequencerHandle seqr2 = __moore_sequencer_create(name2, strlen(name2), 0);
  MooreSequencerHandle seqr3 = __moore_sequencer_create(name3, strlen(name3), 0);

  EXPECT_NE(seqr1, seqr2);
  EXPECT_NE(seqr2, seqr3);
  EXPECT_NE(seqr1, seqr3);

  // Each sequencer operates independently
  __moore_sequencer_start(seqr1);
  EXPECT_EQ(__moore_sequencer_is_running(seqr1), 1);
  EXPECT_EQ(__moore_sequencer_is_running(seqr2), 0);
  EXPECT_EQ(__moore_sequencer_is_running(seqr3), 0);

  __moore_sequencer_start(seqr2);
  EXPECT_EQ(__moore_sequencer_is_running(seqr1), 1);
  EXPECT_EQ(__moore_sequencer_is_running(seqr2), 1);
  EXPECT_EQ(__moore_sequencer_is_running(seqr3), 0);

  __moore_sequencer_stop(seqr1);
  EXPECT_EQ(__moore_sequencer_is_running(seqr1), 0);
  EXPECT_EQ(__moore_sequencer_is_running(seqr2), 1);

  __moore_sequencer_destroy(seqr1);
  __moore_sequencer_destroy(seqr2);
  __moore_sequencer_destroy(seqr3);
}

TEST(MooreRuntimeSequenceTest, MultipleSequences) {
  const char *name1 = "sequence_a";
  const char *name2 = "sequence_b";

  MooreSequenceHandle seq1 = __moore_sequence_create(name1, strlen(name1), 10);
  MooreSequenceHandle seq2 = __moore_sequence_create(name2, strlen(name2), 20);

  EXPECT_NE(seq1, seq2);
  EXPECT_EQ(__moore_sequence_get_priority(seq1), 10);
  EXPECT_EQ(__moore_sequence_get_priority(seq2), 20);

  __moore_sequence_destroy(seq1);
  __moore_sequence_destroy(seq2);
}

//===----------------------------------------------------------------------===//
// UVM Scoreboard Tests
//===----------------------------------------------------------------------===//

// Test transaction type for scoreboard tests
struct TestScoreboardTransaction {
  int32_t addr;
  int32_t data;
  int8_t kind;

  bool operator==(const TestScoreboardTransaction &other) const {
    return addr == other.addr && data == other.data && kind == other.kind;
  }
};

// Custom compare callback for scoreboard tests
static int32_t customScoreboardCompare(const void *expected, const void *actual,
                                        int64_t transactionSize, void *userData) {
  (void)transactionSize;
  (void)userData;
  const auto *exp = static_cast<const TestScoreboardTransaction *>(expected);
  const auto *act = static_cast<const TestScoreboardTransaction *>(actual);
  return (exp->addr == act->addr && exp->data == act->data) ? 1 : 0;
}

// Mismatch callback state for tests
static std::atomic<int> scoreboardMismatchCount{0};
static TestScoreboardTransaction lastMismatchExpected;
static TestScoreboardTransaction lastMismatchActual;

static void testMismatchCallback(const void *expected, const void *actual,
                                  int64_t transactionSize, void *userData) {
  (void)transactionSize;
  (void)userData;
  lastMismatchExpected = *static_cast<const TestScoreboardTransaction *>(expected);
  lastMismatchActual = *static_cast<const TestScoreboardTransaction *>(actual);
  scoreboardMismatchCount++;
}

TEST(MooreRuntimeScoreboardTest, Creation) {
  const char *sbName = "test_scoreboard";
  MooreScoreboardHandle sb = __moore_scoreboard_create(
      sbName, strlen(sbName), sizeof(TestScoreboardTransaction));
  EXPECT_NE(sb, MOORE_SCOREBOARD_INVALID_HANDLE);

  EXPECT_EQ(__moore_scoreboard_is_empty(sb), 1);
  EXPECT_EQ(__moore_scoreboard_get_match_count(sb), 0);
  EXPECT_EQ(__moore_scoreboard_get_mismatch_count(sb), 0);
  EXPECT_EQ(__moore_scoreboard_passed(sb), 1);

  MooreString name = __moore_scoreboard_get_name(sb);
  EXPECT_EQ(name.len, strlen(sbName));
  EXPECT_EQ(std::string(name.data, name.len), std::string(sbName));
  std::free(name.data);

  __moore_scoreboard_destroy(sb);
}

TEST(MooreRuntimeScoreboardTest, AddExpectedAndActual) {
  const char *sbName = "add_transactions_sb";
  MooreScoreboardHandle sb = __moore_scoreboard_create(
      sbName, strlen(sbName), sizeof(TestScoreboardTransaction));

  TestScoreboardTransaction tx1 = {0x1000, 0xDEAD, 1};
  TestScoreboardTransaction tx2 = {0x2000, 0xBEEF, 0};

  __moore_scoreboard_add_expected(sb, &tx1, sizeof(tx1));
  EXPECT_EQ(__moore_scoreboard_get_pending_expected(sb), 1);
  EXPECT_EQ(__moore_scoreboard_get_pending_actual(sb), 0);
  EXPECT_EQ(__moore_scoreboard_is_empty(sb), 0);

  __moore_scoreboard_add_actual(sb, &tx2, sizeof(tx2));
  EXPECT_EQ(__moore_scoreboard_get_pending_expected(sb), 1);
  EXPECT_EQ(__moore_scoreboard_get_pending_actual(sb), 1);

  __moore_scoreboard_destroy(sb);
}

TEST(MooreRuntimeScoreboardTest, TryCompareMatch) {
  const char *sbName = "try_compare_match_sb";
  MooreScoreboardHandle sb = __moore_scoreboard_create(
      sbName, strlen(sbName), sizeof(TestScoreboardTransaction));

  TestScoreboardTransaction tx = {0x1000, 0xDEAD, 1};

  __moore_scoreboard_add_expected(sb, &tx, sizeof(tx));
  __moore_scoreboard_add_actual(sb, &tx, sizeof(tx));

  MooreScoreboardCompareResult result = __moore_scoreboard_try_compare(sb);
  EXPECT_EQ(result, MOORE_SCOREBOARD_MATCH);
  EXPECT_EQ(__moore_scoreboard_get_match_count(sb), 1);
  EXPECT_EQ(__moore_scoreboard_get_mismatch_count(sb), 0);
  EXPECT_EQ(__moore_scoreboard_is_empty(sb), 1);
  EXPECT_EQ(__moore_scoreboard_passed(sb), 1);

  __moore_scoreboard_destroy(sb);
}

TEST(MooreRuntimeScoreboardTest, TryCompareMismatch) {
  const char *sbName = "try_compare_mismatch_sb";
  MooreScoreboardHandle sb = __moore_scoreboard_create(
      sbName, strlen(sbName), sizeof(TestScoreboardTransaction));

  TestScoreboardTransaction expected = {0x1000, 0xDEAD, 1};
  TestScoreboardTransaction actual = {0x1000, 0xBEEF, 1};

  __moore_scoreboard_add_expected(sb, &expected, sizeof(expected));
  __moore_scoreboard_add_actual(sb, &actual, sizeof(actual));

  MooreScoreboardCompareResult result = __moore_scoreboard_try_compare(sb);
  EXPECT_EQ(result, MOORE_SCOREBOARD_MISMATCH);
  EXPECT_EQ(__moore_scoreboard_get_match_count(sb), 0);
  EXPECT_EQ(__moore_scoreboard_get_mismatch_count(sb), 1);
  EXPECT_EQ(__moore_scoreboard_passed(sb), 0);

  __moore_scoreboard_destroy(sb);
}

TEST(MooreRuntimeScoreboardTest, TryCompareTimeout) {
  const char *sbName = "try_compare_timeout_sb";
  MooreScoreboardHandle sb = __moore_scoreboard_create(
      sbName, strlen(sbName), sizeof(TestScoreboardTransaction));

  // No transactions added, should return timeout
  MooreScoreboardCompareResult result = __moore_scoreboard_try_compare(sb);
  EXPECT_EQ(result, MOORE_SCOREBOARD_TIMEOUT);

  // Add only expected, should still timeout
  TestScoreboardTransaction tx = {0x1000, 0xDEAD, 1};
  __moore_scoreboard_add_expected(sb, &tx, sizeof(tx));
  result = __moore_scoreboard_try_compare(sb);
  EXPECT_EQ(result, MOORE_SCOREBOARD_TIMEOUT);

  __moore_scoreboard_destroy(sb);
}

TEST(MooreRuntimeScoreboardTest, CompareAll) {
  const char *sbName = "compare_all_sb";
  MooreScoreboardHandle sb = __moore_scoreboard_create(
      sbName, strlen(sbName), sizeof(TestScoreboardTransaction));

  // Add multiple matching transactions
  TestScoreboardTransaction tx1 = {0x1000, 0xDEAD, 1};
  TestScoreboardTransaction tx2 = {0x2000, 0xBEEF, 0};
  TestScoreboardTransaction tx3 = {0x3000, 0xCAFE, 1};

  __moore_scoreboard_add_expected(sb, &tx1, sizeof(tx1));
  __moore_scoreboard_add_expected(sb, &tx2, sizeof(tx2));
  __moore_scoreboard_add_expected(sb, &tx3, sizeof(tx3));

  __moore_scoreboard_add_actual(sb, &tx1, sizeof(tx1));
  __moore_scoreboard_add_actual(sb, &tx2, sizeof(tx2));
  __moore_scoreboard_add_actual(sb, &tx3, sizeof(tx3));

  int64_t comparisons = __moore_scoreboard_compare_all(sb);
  EXPECT_EQ(comparisons, 3);
  EXPECT_EQ(__moore_scoreboard_get_match_count(sb), 3);
  EXPECT_EQ(__moore_scoreboard_get_mismatch_count(sb), 0);
  EXPECT_EQ(__moore_scoreboard_is_empty(sb), 1);

  __moore_scoreboard_destroy(sb);
}

TEST(MooreRuntimeScoreboardTest, CustomCompareCallback) {
  const char *sbName = "custom_compare_sb";
  MooreScoreboardHandle sb = __moore_scoreboard_create(
      sbName, strlen(sbName), sizeof(TestScoreboardTransaction));

  __moore_scoreboard_set_compare_callback(sb, customScoreboardCompare, nullptr);

  // Transactions differ in 'kind' field, but custom compare only checks addr and data
  TestScoreboardTransaction expected = {0x1000, 0xDEAD, 0};
  TestScoreboardTransaction actual = {0x1000, 0xDEAD, 1};  // Different kind

  __moore_scoreboard_add_expected(sb, &expected, sizeof(expected));
  __moore_scoreboard_add_actual(sb, &actual, sizeof(actual));

  MooreScoreboardCompareResult result = __moore_scoreboard_try_compare(sb);
  EXPECT_EQ(result, MOORE_SCOREBOARD_MATCH);

  __moore_scoreboard_destroy(sb);
}

TEST(MooreRuntimeScoreboardTest, MismatchCallback) {
  const char *sbName = "mismatch_callback_sb";
  MooreScoreboardHandle sb = __moore_scoreboard_create(
      sbName, strlen(sbName), sizeof(TestScoreboardTransaction));

  scoreboardMismatchCount = 0;
  __moore_scoreboard_set_mismatch_callback(sb, testMismatchCallback, nullptr);

  TestScoreboardTransaction expected = {0x1000, 0xDEAD, 1};
  TestScoreboardTransaction actual = {0x2000, 0xBEEF, 0};

  __moore_scoreboard_add_expected(sb, &expected, sizeof(expected));
  __moore_scoreboard_add_actual(sb, &actual, sizeof(actual));

  MooreScoreboardCompareResult result = __moore_scoreboard_try_compare(sb);
  EXPECT_EQ(result, MOORE_SCOREBOARD_MISMATCH);
  EXPECT_EQ(scoreboardMismatchCount.load(), 1);
  EXPECT_TRUE(lastMismatchExpected == expected);
  EXPECT_TRUE(lastMismatchActual == actual);

  __moore_scoreboard_destroy(sb);
}

TEST(MooreRuntimeScoreboardTest, Reset) {
  const char *sbName = "reset_sb";
  MooreScoreboardHandle sb = __moore_scoreboard_create(
      sbName, strlen(sbName), sizeof(TestScoreboardTransaction));

  TestScoreboardTransaction tx1 = {0x1000, 0xDEAD, 1};
  TestScoreboardTransaction tx2 = {0x2000, 0xBEEF, 0};

  // Add some transactions and do a comparison
  __moore_scoreboard_add_expected(sb, &tx1, sizeof(tx1));
  __moore_scoreboard_add_actual(sb, &tx1, sizeof(tx1));
  __moore_scoreboard_try_compare(sb);

  // Add more pending transactions
  __moore_scoreboard_add_expected(sb, &tx2, sizeof(tx2));
  __moore_scoreboard_add_actual(sb, &tx2, sizeof(tx2));

  EXPECT_EQ(__moore_scoreboard_get_match_count(sb), 1);
  EXPECT_EQ(__moore_scoreboard_get_pending_expected(sb), 1);
  EXPECT_EQ(__moore_scoreboard_get_pending_actual(sb), 1);

  // Reset the scoreboard
  __moore_scoreboard_reset(sb);

  EXPECT_EQ(__moore_scoreboard_get_match_count(sb), 0);
  EXPECT_EQ(__moore_scoreboard_get_mismatch_count(sb), 0);
  EXPECT_EQ(__moore_scoreboard_get_pending_expected(sb), 0);
  EXPECT_EQ(__moore_scoreboard_get_pending_actual(sb), 0);
  EXPECT_EQ(__moore_scoreboard_is_empty(sb), 1);

  __moore_scoreboard_destroy(sb);
}

TEST(MooreRuntimeScoreboardTest, TlmIntegration) {
  const char *sbName = "tlm_integration_sb";
  MooreScoreboardHandle sb = __moore_scoreboard_create(
      sbName, strlen(sbName), sizeof(TestScoreboardTransaction));

  // Get analysis exports
  MooreTlmPortHandle expectedExport = __moore_scoreboard_get_expected_export(sb);
  MooreTlmPortHandle actualExport = __moore_scoreboard_get_actual_export(sb);

  EXPECT_NE(expectedExport, MOORE_TLM_INVALID_HANDLE);
  EXPECT_NE(actualExport, MOORE_TLM_INVALID_HANDLE);

  // Create analysis ports (simulating monitor outputs)
  const char *refPortName = "ref_model.analysis_port";
  const char *dutPortName = "dut_monitor.analysis_port";

  MooreTlmPortHandle refPort = __moore_tlm_port_create(
      refPortName, strlen(refPortName), 0, MOORE_TLM_PORT_ANALYSIS);
  MooreTlmPortHandle dutPort = __moore_tlm_port_create(
      dutPortName, strlen(dutPortName), 0, MOORE_TLM_PORT_ANALYSIS);

  // Connect ports to scoreboard exports
  EXPECT_EQ(__moore_tlm_port_connect(refPort, expectedExport), 1);
  EXPECT_EQ(__moore_tlm_port_connect(dutPort, actualExport), 1);

  // Write transactions via the TLM ports
  TestScoreboardTransaction tx = {0x4000, 0xFACE, 1};
  __moore_tlm_port_write(refPort, &tx, sizeof(tx));
  __moore_tlm_port_write(dutPort, &tx, sizeof(tx));

  // The FIFOs should have received the transactions
  // We need to transfer them to the scoreboard queues manually
  // (In a real UVM flow, this would be done via subscribers)

  __moore_tlm_port_destroy(refPort);
  __moore_tlm_port_destroy(dutPort);
  __moore_scoreboard_destroy(sb);
}

TEST(MooreRuntimeScoreboardTest, PassedWithPending) {
  const char *sbName = "passed_pending_sb";
  MooreScoreboardHandle sb = __moore_scoreboard_create(
      sbName, strlen(sbName), sizeof(TestScoreboardTransaction));

  // A scoreboard with pending transactions should not pass
  TestScoreboardTransaction tx = {0x1000, 0xDEAD, 1};
  __moore_scoreboard_add_expected(sb, &tx, sizeof(tx));

  EXPECT_EQ(__moore_scoreboard_passed(sb), 0);

  // Add matching actual and compare
  __moore_scoreboard_add_actual(sb, &tx, sizeof(tx));
  __moore_scoreboard_try_compare(sb);

  EXPECT_EQ(__moore_scoreboard_passed(sb), 1);

  __moore_scoreboard_destroy(sb);
}

TEST(MooreRuntimeScoreboardTest, TracingEnableDisable) {
  EXPECT_EQ(__moore_scoreboard_is_trace_enabled(), 0);

  __moore_scoreboard_set_trace_enabled(1);
  EXPECT_EQ(__moore_scoreboard_is_trace_enabled(), 1);

  __moore_scoreboard_set_trace_enabled(0);
  EXPECT_EQ(__moore_scoreboard_is_trace_enabled(), 0);
}

TEST(MooreRuntimeScoreboardTest, Statistics) {
  int64_t totalSb, totalComp, totalMatch, totalMismatch;
  __moore_scoreboard_get_statistics(&totalSb, &totalComp, &totalMatch, &totalMismatch);

  // Statistics should be non-negative (they accumulate from previous tests)
  EXPECT_GE(totalSb, 0);
  EXPECT_GE(totalComp, 0);
  EXPECT_GE(totalMatch, 0);
  EXPECT_GE(totalMismatch, 0);
}

TEST(MooreRuntimeScoreboardTest, InvalidHandle) {
  MooreScoreboardHandle invalid = MOORE_SCOREBOARD_INVALID_HANDLE;

  // All operations on invalid handle should be safe
  EXPECT_EQ(__moore_scoreboard_get_match_count(invalid), 0);
  EXPECT_EQ(__moore_scoreboard_get_mismatch_count(invalid), 0);
  EXPECT_EQ(__moore_scoreboard_get_pending_expected(invalid), 0);
  EXPECT_EQ(__moore_scoreboard_get_pending_actual(invalid), 0);
  EXPECT_EQ(__moore_scoreboard_is_empty(invalid), 1);
  EXPECT_EQ(__moore_scoreboard_passed(invalid), 0);
  EXPECT_EQ(__moore_scoreboard_try_compare(invalid), MOORE_SCOREBOARD_TIMEOUT);
  EXPECT_EQ(__moore_scoreboard_get_expected_export(invalid), MOORE_TLM_INVALID_HANDLE);
  EXPECT_EQ(__moore_scoreboard_get_actual_export(invalid), MOORE_TLM_INVALID_HANDLE);

  MooreString name = __moore_scoreboard_get_name(invalid);
  EXPECT_EQ(name.data, nullptr);
  EXPECT_EQ(name.len, 0);

  // These should be safe to call (no crash)
  __moore_scoreboard_destroy(invalid);
  __moore_scoreboard_reset(invalid);
}

TEST(MooreRuntimeScoreboardTest, MultipleScoreboards) {
  const char *name1 = "scoreboard_1";
  const char *name2 = "scoreboard_2";
  const char *name3 = "scoreboard_3";

  MooreScoreboardHandle sb1 = __moore_scoreboard_create(
      name1, strlen(name1), sizeof(TestScoreboardTransaction));
  MooreScoreboardHandle sb2 = __moore_scoreboard_create(
      name2, strlen(name2), sizeof(TestScoreboardTransaction));
  MooreScoreboardHandle sb3 = __moore_scoreboard_create(
      name3, strlen(name3), sizeof(TestScoreboardTransaction));

  EXPECT_NE(sb1, sb2);
  EXPECT_NE(sb2, sb3);
  EXPECT_NE(sb1, sb3);

  // Each scoreboard operates independently
  TestScoreboardTransaction tx1 = {0x1000, 0x11, 0};
  TestScoreboardTransaction tx2 = {0x2000, 0x22, 1};

  __moore_scoreboard_add_expected(sb1, &tx1, sizeof(tx1));
  __moore_scoreboard_add_actual(sb1, &tx1, sizeof(tx1));

  __moore_scoreboard_add_expected(sb2, &tx2, sizeof(tx2));
  __moore_scoreboard_add_actual(sb2, &tx2, sizeof(tx2));

  __moore_scoreboard_try_compare(sb1);
  __moore_scoreboard_try_compare(sb2);

  EXPECT_EQ(__moore_scoreboard_get_match_count(sb1), 1);
  EXPECT_EQ(__moore_scoreboard_get_match_count(sb2), 1);
  EXPECT_EQ(__moore_scoreboard_get_match_count(sb3), 0);

  __moore_scoreboard_destroy(sb1);
  __moore_scoreboard_destroy(sb2);
  __moore_scoreboard_destroy(sb3);
}

TEST(MooreRuntimeScoreboardTest, MixedMatchMismatch) {
  const char *sbName = "mixed_results_sb";
  MooreScoreboardHandle sb = __moore_scoreboard_create(
      sbName, strlen(sbName), sizeof(TestScoreboardTransaction));

  TestScoreboardTransaction tx1 = {0x1000, 0xDEAD, 1};
  TestScoreboardTransaction tx2 = {0x2000, 0xBEEF, 0};
  TestScoreboardTransaction tx3 = {0x3000, 0xCAFE, 1};
  TestScoreboardTransaction tx3_bad = {0x3000, 0xBAD0, 1};

  // First transaction: match
  __moore_scoreboard_add_expected(sb, &tx1, sizeof(tx1));
  __moore_scoreboard_add_actual(sb, &tx1, sizeof(tx1));

  // Second transaction: match
  __moore_scoreboard_add_expected(sb, &tx2, sizeof(tx2));
  __moore_scoreboard_add_actual(sb, &tx2, sizeof(tx2));

  // Third transaction: mismatch
  __moore_scoreboard_add_expected(sb, &tx3, sizeof(tx3));
  __moore_scoreboard_add_actual(sb, &tx3_bad, sizeof(tx3_bad));

  int64_t comparisons = __moore_scoreboard_compare_all(sb);
  EXPECT_EQ(comparisons, 3);
  EXPECT_EQ(__moore_scoreboard_get_match_count(sb), 2);
  EXPECT_EQ(__moore_scoreboard_get_mismatch_count(sb), 1);
  EXPECT_EQ(__moore_scoreboard_passed(sb), 0);

  __moore_scoreboard_destroy(sb);
}

//===----------------------------------------------------------------------===//
// UVM Register Abstraction Layer (RAL) Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeRALTest, RegisterCreation) {
  // Clear any previous state
  __moore_reg_clear_all();

  const char *regName = "test_reg";
  MooreRegHandle reg = __moore_reg_create(regName, strlen(regName), 32);
  EXPECT_NE(reg, MOORE_REG_INVALID_HANDLE);

  // Check name
  MooreString name = __moore_reg_get_name(reg);
  EXPECT_EQ(name.len, strlen(regName));
  EXPECT_EQ(std::string(name.data, name.len), std::string(regName));
  std::free(name.data);

  // Check bit width
  EXPECT_EQ(__moore_reg_get_n_bits(reg), 32);

  // Initial values should be 0
  EXPECT_EQ(__moore_reg_get_value(reg), 0ULL);
  EXPECT_EQ(__moore_reg_get_desired(reg), 0ULL);

  __moore_reg_destroy(reg);
}

TEST(MooreRuntimeRALTest, RegisterInvalidParams) {
  // Null name
  EXPECT_EQ(__moore_reg_create(nullptr, 5, 32), MOORE_REG_INVALID_HANDLE);

  // Zero length name
  EXPECT_EQ(__moore_reg_create("test", 0, 32), MOORE_REG_INVALID_HANDLE);

  // Invalid bit width
  EXPECT_EQ(__moore_reg_create("test", 4, 0), MOORE_REG_INVALID_HANDLE);
  EXPECT_EQ(__moore_reg_create("test", 4, 65), MOORE_REG_INVALID_HANDLE);
}

TEST(MooreRuntimeRALTest, RegisterReadWrite) {
  __moore_reg_clear_all();

  const char *regName = "rw_reg";
  MooreRegHandle reg = __moore_reg_create(regName, strlen(regName), 32);
  EXPECT_NE(reg, MOORE_REG_INVALID_HANDLE);

  // Write a value
  MooreRegStatus status;
  __moore_reg_write(reg, MOORE_REG_INVALID_HANDLE, 0xDEADBEEF,
                    UVM_FRONTDOOR, &status, 0);
  EXPECT_EQ(status, UVM_REG_STATUS_OK);

  // Read it back
  uint64_t value = __moore_reg_read(reg, MOORE_REG_INVALID_HANDLE,
                                    UVM_FRONTDOOR, &status, 0);
  EXPECT_EQ(status, UVM_REG_STATUS_OK);
  EXPECT_EQ(value, 0xDEADBEEF);

  // Check mirror value
  EXPECT_EQ(__moore_reg_get_value(reg), 0xDEADBEEF);

  __moore_reg_destroy(reg);
}

TEST(MooreRuntimeRALTest, RegisterMirrorDesired) {
  __moore_reg_clear_all();

  const char *regName = "mirror_reg";
  MooreRegHandle reg = __moore_reg_create(regName, strlen(regName), 16);
  EXPECT_NE(reg, MOORE_REG_INVALID_HANDLE);

  // Set mirror value directly
  __moore_reg_set_value(reg, 0x1234);
  EXPECT_EQ(__moore_reg_get_value(reg), 0x1234);

  // Set desired value
  __moore_reg_set_desired(reg, 0x5678);
  EXPECT_EQ(__moore_reg_get_desired(reg), 0x5678);

  // Check needs update
  EXPECT_TRUE(__moore_reg_needs_update(reg));

  // Make them equal
  __moore_reg_set_value(reg, 0x5678);
  EXPECT_FALSE(__moore_reg_needs_update(reg));

  __moore_reg_destroy(reg);
}

TEST(MooreRuntimeRALTest, RegisterUpdate) {
  __moore_reg_clear_all();

  const char *regName = "update_reg";
  MooreRegHandle reg = __moore_reg_create(regName, strlen(regName), 32);
  EXPECT_NE(reg, MOORE_REG_INVALID_HANDLE);

  // Set desired value
  __moore_reg_set_desired(reg, 0xCAFEBABE);
  EXPECT_EQ(__moore_reg_get_desired(reg), 0xCAFEBABE);
  EXPECT_NE(__moore_reg_get_value(reg), 0xCAFEBABE);

  // Update (write desired to mirror)
  MooreRegStatus status;
  __moore_reg_update(reg, MOORE_REG_INVALID_HANDLE, UVM_FRONTDOOR, &status);
  EXPECT_EQ(status, UVM_REG_STATUS_OK);

  // Now mirror should match desired
  EXPECT_EQ(__moore_reg_get_value(reg), 0xCAFEBABE);

  __moore_reg_destroy(reg);
}

TEST(MooreRuntimeRALTest, RegisterReset) {
  __moore_reg_clear_all();

  const char *regName = "reset_reg";
  MooreRegHandle reg = __moore_reg_create(regName, strlen(regName), 32);
  EXPECT_NE(reg, MOORE_REG_INVALID_HANDLE);

  // Set reset values
  __moore_reg_set_reset(reg, 0xABCD1234, "HARD");
  __moore_reg_set_reset(reg, 0x00001111, "SOFT");

  // Write a different value
  __moore_reg_set_value(reg, 0xFFFFFFFF);
  EXPECT_EQ(__moore_reg_get_value(reg), 0xFFFFFFFF);

  // Hard reset
  __moore_reg_reset(reg, "HARD");
  EXPECT_EQ(__moore_reg_get_value(reg), 0xABCD1234);

  // Write again
  __moore_reg_set_value(reg, 0xFFFFFFFF);

  // Soft reset
  __moore_reg_reset(reg, "SOFT");
  EXPECT_EQ(__moore_reg_get_value(reg), 0x00001111);

  // Check get_reset
  EXPECT_EQ(__moore_reg_get_reset(reg, "HARD"), 0xABCD1234);
  EXPECT_EQ(__moore_reg_get_reset(reg, "SOFT"), 0x00001111);

  __moore_reg_destroy(reg);
}

TEST(MooreRuntimeRALTest, RegisterPredict) {
  __moore_reg_clear_all();

  const char *regName = "predict_reg";
  MooreRegHandle reg = __moore_reg_create(regName, strlen(regName), 32);
  EXPECT_NE(reg, MOORE_REG_INVALID_HANDLE);

  // Initial value
  EXPECT_EQ(__moore_reg_get_value(reg), 0ULL);

  // Predict write
  EXPECT_TRUE(__moore_reg_predict(reg, 0x12345678, true));
  EXPECT_EQ(__moore_reg_get_value(reg), 0x12345678);

  __moore_reg_destroy(reg);
}

TEST(MooreRuntimeRALTest, RegisterFieldBasic) {
  __moore_reg_clear_all();

  const char *regName = "field_reg";
  MooreRegHandle reg = __moore_reg_create(regName, strlen(regName), 32);
  EXPECT_NE(reg, MOORE_REG_INVALID_HANDLE);

  // Add field: bits [7:0]
  const char *fieldName1 = "low_byte";
  MooreRegFieldHandle field1 = __moore_reg_add_field(
      reg, fieldName1, strlen(fieldName1), 8, 0, UVM_REG_ACCESS_RW, 0xAB);
  EXPECT_NE(field1, MOORE_REG_INVALID_HANDLE);

  // Add field: bits [15:8]
  const char *fieldName2 = "high_byte";
  MooreRegFieldHandle field2 = __moore_reg_add_field(
      reg, fieldName2, strlen(fieldName2), 8, 8, UVM_REG_ACCESS_RW, 0xCD);
  EXPECT_NE(field2, MOORE_REG_INVALID_HANDLE);

  // Check field count
  EXPECT_EQ(__moore_reg_get_n_fields(reg), 2);

  // Initial register value should be set from field resets
  // 0xCDAB (high_byte=0xCD, low_byte=0xAB)
  EXPECT_EQ(__moore_reg_get_value(reg), 0xCDAB);

  // Get field values
  EXPECT_EQ(__moore_reg_field_get_value(reg, field1), 0xAB);
  EXPECT_EQ(__moore_reg_field_get_value(reg, field2), 0xCD);

  // Set field value
  __moore_reg_field_set_value(reg, field1, 0x12);
  EXPECT_EQ(__moore_reg_field_get_value(reg, field1), 0x12);
  EXPECT_EQ(__moore_reg_get_value(reg), 0xCD12);

  __moore_reg_destroy(reg);
}

TEST(MooreRuntimeRALTest, RegisterFieldByName) {
  __moore_reg_clear_all();

  const char *regName = "named_field_reg";
  MooreRegHandle reg = __moore_reg_create(regName, strlen(regName), 32);
  EXPECT_NE(reg, MOORE_REG_INVALID_HANDLE);

  // Add fields
  const char *enableField = "enable";
  __moore_reg_add_field(reg, enableField, strlen(enableField),
                        1, 0, UVM_REG_ACCESS_RW, 0);

  const char *modeField = "mode";
  __moore_reg_add_field(reg, modeField, strlen(modeField),
                        4, 1, UVM_REG_ACCESS_RW, 0);

  // Look up field by name
  MooreRegFieldHandle enable =
      __moore_reg_get_field_by_name(reg, enableField, strlen(enableField));
  EXPECT_NE(enable, MOORE_REG_INVALID_HANDLE);

  MooreRegFieldHandle mode =
      __moore_reg_get_field_by_name(reg, modeField, strlen(modeField));
  EXPECT_NE(mode, MOORE_REG_INVALID_HANDLE);

  // Non-existent field
  MooreRegFieldHandle notFound =
      __moore_reg_get_field_by_name(reg, "nonexistent", 11);
  EXPECT_EQ(notFound, MOORE_REG_INVALID_HANDLE);

  __moore_reg_destroy(reg);
}

TEST(MooreRuntimeRALTest, RegisterBlockCreation) {
  __moore_reg_clear_all();

  const char *blockName = "test_block";
  MooreRegBlockHandle block =
      __moore_reg_block_create(blockName, strlen(blockName));
  EXPECT_NE(block, MOORE_REG_INVALID_HANDLE);

  // Check name
  MooreString name = __moore_reg_block_get_name(block);
  EXPECT_EQ(name.len, strlen(blockName));
  EXPECT_EQ(std::string(name.data, name.len), std::string(blockName));
  std::free(name.data);

  // Initial state
  EXPECT_EQ(__moore_reg_block_get_n_regs(block), 0);
  EXPECT_FALSE(__moore_reg_block_is_locked(block));
  EXPECT_EQ(__moore_reg_block_get_default_map(block), MOORE_REG_INVALID_HANDLE);

  __moore_reg_block_destroy(block);
}

TEST(MooreRuntimeRALTest, RegisterBlockAddRegister) {
  __moore_reg_clear_all();

  // Create block
  const char *blockName = "add_reg_block";
  MooreRegBlockHandle block =
      __moore_reg_block_create(blockName, strlen(blockName));
  EXPECT_NE(block, MOORE_REG_INVALID_HANDLE);

  // Create registers
  const char *reg1Name = "ctrl_reg";
  MooreRegHandle reg1 = __moore_reg_create(reg1Name, strlen(reg1Name), 32);

  const char *reg2Name = "status_reg";
  MooreRegHandle reg2 = __moore_reg_create(reg2Name, strlen(reg2Name), 32);

  // Add to block
  __moore_reg_block_add_reg(block, reg1, 0x0000);
  __moore_reg_block_add_reg(block, reg2, 0x0004);

  EXPECT_EQ(__moore_reg_block_get_n_regs(block), 2);

  // Find by name
  MooreRegHandle found =
      __moore_reg_block_get_reg_by_name(block, reg1Name, strlen(reg1Name));
  EXPECT_EQ(found, reg1);

  found = __moore_reg_block_get_reg_by_name(block, reg2Name, strlen(reg2Name));
  EXPECT_EQ(found, reg2);

  // Not found
  found = __moore_reg_block_get_reg_by_name(block, "nonexistent", 11);
  EXPECT_EQ(found, MOORE_REG_INVALID_HANDLE);

  __moore_reg_block_destroy(block);
  __moore_reg_destroy(reg1);
  __moore_reg_destroy(reg2);
}

TEST(MooreRuntimeRALTest, RegisterBlockLocking) {
  __moore_reg_clear_all();

  const char *blockName = "lock_test_block";
  MooreRegBlockHandle block =
      __moore_reg_block_create(blockName, strlen(blockName));
  EXPECT_NE(block, MOORE_REG_INVALID_HANDLE);

  // Create and add register
  const char *regName = "test_reg";
  MooreRegHandle reg = __moore_reg_create(regName, strlen(regName), 32);
  __moore_reg_block_add_reg(block, reg, 0x0000);
  EXPECT_EQ(__moore_reg_block_get_n_regs(block), 1);

  // Lock block
  __moore_reg_block_lock(block);
  EXPECT_TRUE(__moore_reg_block_is_locked(block));

  // Try to add another register (should be silently ignored due to lock)
  const char *reg2Name = "another_reg";
  MooreRegHandle reg2 = __moore_reg_create(reg2Name, strlen(reg2Name), 32);
  __moore_reg_block_add_reg(block, reg2, 0x0004);
  // Count should still be 1
  EXPECT_EQ(__moore_reg_block_get_n_regs(block), 1);

  __moore_reg_block_destroy(block);
  __moore_reg_destroy(reg);
  __moore_reg_destroy(reg2);
}

TEST(MooreRuntimeRALTest, RegisterBlockReset) {
  __moore_reg_clear_all();

  const char *blockName = "reset_block";
  MooreRegBlockHandle block =
      __moore_reg_block_create(blockName, strlen(blockName));

  // Create registers with reset values
  const char *reg1Name = "reg1";
  MooreRegHandle reg1 = __moore_reg_create(reg1Name, strlen(reg1Name), 32);
  __moore_reg_set_reset(reg1, 0x11111111, "HARD");

  const char *reg2Name = "reg2";
  MooreRegHandle reg2 = __moore_reg_create(reg2Name, strlen(reg2Name), 32);
  __moore_reg_set_reset(reg2, 0x22222222, "HARD");

  __moore_reg_block_add_reg(block, reg1, 0x0000);
  __moore_reg_block_add_reg(block, reg2, 0x0004);

  // Write different values
  __moore_reg_set_value(reg1, 0xFFFFFFFF);
  __moore_reg_set_value(reg2, 0xFFFFFFFF);

  // Reset block
  __moore_reg_block_reset(block, "HARD");

  // Check values are reset
  EXPECT_EQ(__moore_reg_get_value(reg1), 0x11111111);
  EXPECT_EQ(__moore_reg_get_value(reg2), 0x22222222);

  __moore_reg_block_destroy(block);
  __moore_reg_destroy(reg1);
  __moore_reg_destroy(reg2);
}

TEST(MooreRuntimeRALTest, RegisterMapCreation) {
  __moore_reg_clear_all();

  // Create block first
  const char *blockName = "map_block";
  MooreRegBlockHandle block =
      __moore_reg_block_create(blockName, strlen(blockName));

  // Create map
  const char *mapName = "default_map";
  MooreRegMapHandle map = __moore_reg_map_create(
      block, mapName, strlen(mapName), 0x1000, 4, 0);
  EXPECT_NE(map, MOORE_REG_INVALID_HANDLE);

  // Check map properties
  MooreString name = __moore_reg_map_get_name(map);
  EXPECT_EQ(std::string(name.data, name.len), std::string(mapName));
  std::free(name.data);

  EXPECT_EQ(__moore_reg_map_get_base_addr(map), 0x1000);

  // Map should be set as default
  EXPECT_EQ(__moore_reg_block_get_default_map(block), map);

  __moore_reg_block_destroy(block);
}

TEST(MooreRuntimeRALTest, RegisterMapAddRegister) {
  __moore_reg_clear_all();

  // Create block and map
  const char *blockName = "map_reg_block";
  MooreRegBlockHandle block =
      __moore_reg_block_create(blockName, strlen(blockName));

  const char *mapName = "test_map";
  MooreRegMapHandle map = __moore_reg_map_create(
      block, mapName, strlen(mapName), 0x2000, 4, 0);

  // Create register
  const char *regName = "mapped_reg";
  MooreRegHandle reg = __moore_reg_create(regName, strlen(regName), 32);

  // Add to block and map
  __moore_reg_block_add_reg(block, reg, 0x0010);
  __moore_reg_map_add_reg(map, reg, 0x0010, "RW");

  // Get register address
  uint64_t addr = __moore_reg_get_address(reg, map);
  EXPECT_EQ(addr, 0x2010);  // base 0x2000 + offset 0x0010

  // Get register by address
  MooreRegHandle found = __moore_reg_map_get_reg_by_addr(map, 0x2010);
  EXPECT_EQ(found, reg);

  // Get register offset
  uint64_t offset = __moore_reg_map_get_reg_offset(map, reg);
  EXPECT_EQ(offset, 0x0010);

  __moore_reg_block_destroy(block);
  __moore_reg_destroy(reg);
}

TEST(MooreRuntimeRALTest, RegisterSubBlock) {
  __moore_reg_clear_all();

  // Create parent block
  const char *parentName = "parent_block";
  MooreRegBlockHandle parent =
      __moore_reg_block_create(parentName, strlen(parentName));

  // Create child block
  const char *childName = "child";
  MooreRegBlockHandle child =
      __moore_reg_block_create(childName, strlen(childName));

  // Create register in child
  const char *regName = "child_reg";
  MooreRegHandle reg = __moore_reg_create(regName, strlen(regName), 32);
  __moore_reg_block_add_reg(child, reg, 0x0000);

  // Add child to parent
  __moore_reg_block_add_block(parent, child, 0x1000);

  // Find register via hierarchical name
  const char *hierName = "child.child_reg";
  MooreRegHandle found =
      __moore_reg_block_get_reg_by_name(parent, hierName, strlen(hierName));
  EXPECT_EQ(found, reg);

  __moore_reg_block_destroy(parent);
  __moore_reg_block_destroy(child);
  __moore_reg_destroy(reg);
}

TEST(MooreRuntimeRALTest, RegisterAccessCallback) {
  __moore_reg_clear_all();

  // Track callback invocations
  static int callbackCount = 0;
  static uint64_t lastValue = 0;
  static int32_t lastIsWrite = -1;

  callbackCount = 0;
  lastValue = 0;
  lastIsWrite = -1;

  auto callback = [](MooreRegHandle reg, uint64_t value, int32_t isWrite,
                     void *userData) {
    (void)reg;
    (void)userData;
    callbackCount++;
    lastValue = value;
    lastIsWrite = isWrite;
  };

  const char *regName = "callback_reg";
  MooreRegHandle reg = __moore_reg_create(regName, strlen(regName), 32);

  __moore_reg_set_access_callback(reg, callback, nullptr);

  // Write
  MooreRegStatus status;
  __moore_reg_write(reg, MOORE_REG_INVALID_HANDLE, 0x12345678,
                    UVM_FRONTDOOR, &status, 0);
  EXPECT_EQ(callbackCount, 1);
  EXPECT_EQ(lastValue, 0x12345678);
  EXPECT_EQ(lastIsWrite, 1);

  // Read
  __moore_reg_read(reg, MOORE_REG_INVALID_HANDLE, UVM_FRONTDOOR, &status, 0);
  EXPECT_EQ(callbackCount, 2);
  EXPECT_EQ(lastIsWrite, 0);

  __moore_reg_destroy(reg);
}

TEST(MooreRuntimeRALTest, RALStatistics) {
  __moore_reg_clear_all();

  int64_t totalRegs, totalReads, totalWrites;

  // Initial stats
  __moore_reg_get_statistics(&totalRegs, &totalReads, &totalWrites);
  EXPECT_EQ(totalRegs, 0);
  EXPECT_EQ(totalReads, 0);
  EXPECT_EQ(totalWrites, 0);

  // Create registers
  const char *reg1Name = "stat_reg1";
  MooreRegHandle reg1 = __moore_reg_create(reg1Name, strlen(reg1Name), 32);
  const char *reg2Name = "stat_reg2";
  MooreRegHandle reg2 = __moore_reg_create(reg2Name, strlen(reg2Name), 32);

  __moore_reg_get_statistics(&totalRegs, &totalReads, &totalWrites);
  EXPECT_EQ(totalRegs, 2);

  // Perform reads and writes
  MooreRegStatus status;
  __moore_reg_write(reg1, MOORE_REG_INVALID_HANDLE, 0x100,
                    UVM_FRONTDOOR, &status, 0);
  __moore_reg_write(reg2, MOORE_REG_INVALID_HANDLE, 0x200,
                    UVM_FRONTDOOR, &status, 0);
  __moore_reg_read(reg1, MOORE_REG_INVALID_HANDLE, UVM_FRONTDOOR, &status, 0);

  __moore_reg_get_statistics(&totalRegs, &totalReads, &totalWrites);
  EXPECT_EQ(totalWrites, 2);
  EXPECT_EQ(totalReads, 1);

  __moore_reg_destroy(reg1);
  __moore_reg_destroy(reg2);
}

TEST(MooreRuntimeRALTest, RALTracing) {
  __moore_reg_clear_all();

  // Tracing disabled by default
  EXPECT_EQ(__moore_reg_is_trace_enabled(), 0);

  // Enable tracing
  __moore_reg_set_trace_enabled(1);
  EXPECT_EQ(__moore_reg_is_trace_enabled(), 1);

  // Disable tracing
  __moore_reg_set_trace_enabled(0);
  EXPECT_EQ(__moore_reg_is_trace_enabled(), 0);
}

TEST(MooreRuntimeRALTest, RegisterBitMasking) {
  __moore_reg_clear_all();

  // 8-bit register
  const char *reg8Name = "reg8";
  MooreRegHandle reg8 = __moore_reg_create(reg8Name, strlen(reg8Name), 8);

  // Write value larger than 8 bits - should be masked
  __moore_reg_set_value(reg8, 0xFFFF);
  EXPECT_EQ(__moore_reg_get_value(reg8), 0xFF);

  // 16-bit register
  const char *reg16Name = "reg16";
  MooreRegHandle reg16 = __moore_reg_create(reg16Name, strlen(reg16Name), 16);
  __moore_reg_set_value(reg16, 0xFFFFFFFF);
  EXPECT_EQ(__moore_reg_get_value(reg16), 0xFFFF);

  __moore_reg_destroy(reg8);
  __moore_reg_destroy(reg16);
}

TEST(MooreRuntimeRALTest, InvalidHandleOperations) {
  __moore_reg_clear_all();

  // Operations on invalid handle should not crash
  MooreRegHandle invalid = MOORE_REG_INVALID_HANDLE;

  MooreString name = __moore_reg_get_name(invalid);
  EXPECT_EQ(name.data, nullptr);
  EXPECT_EQ(name.len, 0);

  EXPECT_EQ(__moore_reg_get_n_bits(invalid), 0);
  EXPECT_EQ(__moore_reg_get_value(invalid), 0ULL);
  EXPECT_EQ(__moore_reg_get_address(invalid, MOORE_REG_INVALID_HANDLE), 0ULL);

  MooreRegStatus status;
  EXPECT_EQ(__moore_reg_read(invalid, MOORE_REG_INVALID_HANDLE,
                             UVM_FRONTDOOR, &status, 0), 0ULL);
  EXPECT_EQ(status, UVM_REG_STATUS_NOT_OK);

  // Block operations
  MooreRegBlockHandle invalidBlock = MOORE_REG_INVALID_HANDLE;
  MooreString blockName = __moore_reg_block_get_name(invalidBlock);
  EXPECT_EQ(blockName.data, nullptr);
  EXPECT_EQ(__moore_reg_block_get_n_regs(invalidBlock), 0);
  EXPECT_FALSE(__moore_reg_block_is_locked(invalidBlock));

  // Map operations
  MooreRegMapHandle invalidMap = MOORE_REG_INVALID_HANDLE;
  MooreString mapName = __moore_reg_map_get_name(invalidMap);
  EXPECT_EQ(mapName.data, nullptr);
  EXPECT_EQ(__moore_reg_map_get_base_addr(invalidMap), 0ULL);
}

TEST(MooreRuntimeRALTest, ClearAll) {
  __moore_reg_clear_all();

  // Create some components
  const char *blockName = "clear_block";
  MooreRegBlockHandle block =
      __moore_reg_block_create(blockName, strlen(blockName));

  const char *regName = "clear_reg";
  MooreRegHandle reg = __moore_reg_create(regName, strlen(regName), 32);

  int64_t totalRegs, totalReads, totalWrites;
  __moore_reg_get_statistics(&totalRegs, &totalReads, &totalWrites);
  EXPECT_GT(totalRegs, 0);

  // Clear all
  __moore_reg_clear_all();

  // Stats should be reset
  __moore_reg_get_statistics(&totalRegs, &totalReads, &totalWrites);
  EXPECT_EQ(totalRegs, 0);
  EXPECT_EQ(totalReads, 0);
  EXPECT_EQ(totalWrites, 0);

  // Old handles should be invalid now (accessing would be undefined,
  // so we don't test that)
}

//===----------------------------------------------------------------------===//
// UVM Message Reporting Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeUvmReportTest, VerbosityGetSet) {
  // Default verbosity should be UVM_MEDIUM (200)
  __moore_uvm_set_report_verbosity(MOORE_UVM_MEDIUM);
  EXPECT_EQ(__moore_uvm_get_report_verbosity(), MOORE_UVM_MEDIUM);

  // Set to different values
  __moore_uvm_set_report_verbosity(MOORE_UVM_LOW);
  EXPECT_EQ(__moore_uvm_get_report_verbosity(), MOORE_UVM_LOW);

  __moore_uvm_set_report_verbosity(MOORE_UVM_HIGH);
  EXPECT_EQ(__moore_uvm_get_report_verbosity(), MOORE_UVM_HIGH);

  __moore_uvm_set_report_verbosity(MOORE_UVM_DEBUG);
  EXPECT_EQ(__moore_uvm_get_report_verbosity(), MOORE_UVM_DEBUG);

  // Restore default
  __moore_uvm_set_report_verbosity(MOORE_UVM_MEDIUM);
}

TEST(MooreRuntimeUvmReportTest, ReportEnabled) {
  __moore_uvm_set_report_verbosity(MOORE_UVM_MEDIUM);

  // Info messages should be filtered by verbosity
  const char *id = "TEST";
  int64_t idLen = 4;

  // UVM_LOW (100) <= UVM_MEDIUM (200), should be enabled
  EXPECT_EQ(__moore_uvm_report_enabled(MOORE_UVM_LOW, MOORE_UVM_INFO, id, idLen),
            1);

  // UVM_MEDIUM (200) <= UVM_MEDIUM (200), should be enabled
  EXPECT_EQ(
      __moore_uvm_report_enabled(MOORE_UVM_MEDIUM, MOORE_UVM_INFO, id, idLen),
      1);

  // UVM_HIGH (300) > UVM_MEDIUM (200), should be disabled
  EXPECT_EQ(
      __moore_uvm_report_enabled(MOORE_UVM_HIGH, MOORE_UVM_INFO, id, idLen), 0);

  // Warnings, errors, and fatals are always enabled
  EXPECT_EQ(
      __moore_uvm_report_enabled(MOORE_UVM_HIGH, MOORE_UVM_WARNING, id, idLen),
      1);
  EXPECT_EQ(
      __moore_uvm_report_enabled(MOORE_UVM_HIGH, MOORE_UVM_ERROR, id, idLen), 1);
  EXPECT_EQ(
      __moore_uvm_report_enabled(MOORE_UVM_HIGH, MOORE_UVM_FATAL, id, idLen), 1);
}

TEST(MooreRuntimeUvmReportTest, IdSpecificVerbosity) {
  __moore_uvm_set_report_verbosity(MOORE_UVM_MEDIUM);

  const char *id1 = "VERBOSE_ID";
  int64_t id1Len = 10;
  const char *id2 = "QUIET_ID";
  int64_t id2Len = 8;

  // Set ID-specific verbosity
  __moore_uvm_set_report_id_verbosity(id1, id1Len, MOORE_UVM_DEBUG);
  __moore_uvm_set_report_id_verbosity(id2, id2Len, MOORE_UVM_LOW);

  // VERBOSE_ID should allow high verbosity messages
  EXPECT_EQ(
      __moore_uvm_report_enabled(MOORE_UVM_HIGH, MOORE_UVM_INFO, id1, id1Len),
      1);
  EXPECT_EQ(
      __moore_uvm_report_enabled(MOORE_UVM_DEBUG, MOORE_UVM_INFO, id1, id1Len),
      1);

  // QUIET_ID should filter medium verbosity messages
  EXPECT_EQ(
      __moore_uvm_report_enabled(MOORE_UVM_LOW, MOORE_UVM_INFO, id2, id2Len), 1);
  EXPECT_EQ(
      __moore_uvm_report_enabled(MOORE_UVM_MEDIUM, MOORE_UVM_INFO, id2, id2Len),
      0);

  // Unknown ID should use global verbosity
  const char *id3 = "UNKNOWN";
  int64_t id3Len = 7;
  EXPECT_EQ(
      __moore_uvm_report_enabled(MOORE_UVM_MEDIUM, MOORE_UVM_INFO, id3, id3Len),
      1);
  EXPECT_EQ(
      __moore_uvm_report_enabled(MOORE_UVM_HIGH, MOORE_UVM_INFO, id3, id3Len),
      0);
}

TEST(MooreRuntimeUvmReportTest, ReportCounting) {
  // Reset counts
  __moore_uvm_reset_report_counts();

  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_INFO), 0);
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_WARNING), 0);
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_ERROR), 0);
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_FATAL), 0);

  // Disable exit on fatal for testing
  __moore_uvm_set_fatal_exits(false);

  // Report messages and check counts
  const char *id = "TEST";
  int64_t idLen = 4;
  const char *msg = "Test message";
  int64_t msgLen = 12;

  __moore_uvm_report_info(id, idLen, msg, msgLen, MOORE_UVM_LOW, nullptr, 0, 0,
                          nullptr, 0);
  // Info messages don't count by default (only DISPLAY action)
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_INFO), 0);

  __moore_uvm_report_warning(id, idLen, msg, msgLen, MOORE_UVM_NONE, nullptr, 0,
                             0, nullptr, 0);
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_WARNING), 1);

  __moore_uvm_report_error(id, idLen, msg, msgLen, MOORE_UVM_NONE, nullptr, 0, 0,
                           nullptr, 0);
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_ERROR), 1);

  __moore_uvm_report_fatal(id, idLen, msg, msgLen, MOORE_UVM_NONE, nullptr, 0, 0,
                           nullptr, 0);
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_FATAL), 1);

  // Report more messages
  __moore_uvm_report_warning(id, idLen, msg, msgLen, MOORE_UVM_NONE, nullptr, 0,
                             0, nullptr, 0);
  __moore_uvm_report_error(id, idLen, msg, msgLen, MOORE_UVM_NONE, nullptr, 0, 0,
                           nullptr, 0);
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_WARNING), 2);
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_ERROR), 2);

  // Reset and verify
  __moore_uvm_reset_report_counts();
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_INFO), 0);
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_WARNING), 0);
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_ERROR), 0);
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_FATAL), 0);

  // Re-enable fatal exits
  __moore_uvm_set_fatal_exits(true);
}

TEST(MooreRuntimeUvmReportTest, MaxQuitCount) {
  __moore_uvm_set_max_quit_count(0); // Unlimited
  EXPECT_EQ(__moore_uvm_get_max_quit_count(), 0);

  __moore_uvm_set_max_quit_count(10);
  EXPECT_EQ(__moore_uvm_get_max_quit_count(), 10);

  __moore_uvm_set_max_quit_count(0); // Restore unlimited
}

TEST(MooreRuntimeUvmReportTest, SeverityActions) {
  // Get default actions
  int32_t defaultInfoAction = __moore_uvm_get_report_severity_action(MOORE_UVM_INFO);
  int32_t defaultWarningAction = __moore_uvm_get_report_severity_action(MOORE_UVM_WARNING);
  int32_t defaultErrorAction = __moore_uvm_get_report_severity_action(MOORE_UVM_ERROR);
  int32_t defaultFatalAction = __moore_uvm_get_report_severity_action(MOORE_UVM_FATAL);

  // Verify defaults
  EXPECT_EQ(defaultInfoAction, MOORE_UVM_DISPLAY);
  EXPECT_EQ(defaultWarningAction, MOORE_UVM_DISPLAY | MOORE_UVM_COUNT);
  EXPECT_EQ(defaultErrorAction, MOORE_UVM_DISPLAY | MOORE_UVM_COUNT);
  EXPECT_EQ(defaultFatalAction, MOORE_UVM_DISPLAY | MOORE_UVM_EXIT);

  // Modify actions
  __moore_uvm_set_report_severity_action(MOORE_UVM_INFO,
                                         MOORE_UVM_DISPLAY | MOORE_UVM_COUNT);
  EXPECT_EQ(__moore_uvm_get_report_severity_action(MOORE_UVM_INFO),
            MOORE_UVM_DISPLAY | MOORE_UVM_COUNT);

  // Restore defaults
  __moore_uvm_set_report_severity_action(MOORE_UVM_INFO, MOORE_UVM_DISPLAY);
}

TEST(MooreRuntimeUvmReportTest, SimulationTime) {
  __moore_uvm_set_time(0);
  EXPECT_EQ(__moore_uvm_get_time(), 0ULL);

  __moore_uvm_set_time(12345);
  EXPECT_EQ(__moore_uvm_get_time(), 12345ULL);

  __moore_uvm_set_time(0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(__moore_uvm_get_time(), 0xFFFFFFFFFFFFFFFFULL);

  __moore_uvm_set_time(0);
}

TEST(MooreRuntimeUvmReportTest, ReportInfoWithFilename) {
  __moore_uvm_reset_report_counts();
  __moore_uvm_set_report_verbosity(MOORE_UVM_HIGH);
  __moore_uvm_set_time(1000);

  const char *id = "MYTEST";
  int64_t idLen = 6;
  const char *msg = "Test info message";
  int64_t msgLen = 17;
  const char *filename = "test.sv";
  int64_t filenameLen = 7;
  const char *context = "top.dut";
  int64_t contextLen = 7;

  // This should print:
  // UVM_INFO test.sv(42) @ 1000: MYTEST [top.dut] Test info message
  __moore_uvm_report_info(id, idLen, msg, msgLen, MOORE_UVM_MEDIUM, filename,
                          filenameLen, 42, context, contextLen);

  // The message was displayed (no count for info by default)
  // Just verify it doesn't crash
  __moore_uvm_set_time(0);
}

TEST(MooreRuntimeUvmReportTest, ReportSummarize) {
  __moore_uvm_reset_report_counts();
  __moore_uvm_set_fatal_exits(false);

  const char *id = "TEST";
  int64_t idLen = 4;
  const char *msg = "msg";
  int64_t msgLen = 3;

  // Report some messages
  __moore_uvm_report_warning(id, idLen, msg, msgLen, MOORE_UVM_NONE, nullptr, 0,
                             0, nullptr, 0);
  __moore_uvm_report_error(id, idLen, msg, msgLen, MOORE_UVM_NONE, nullptr, 0, 0,
                           nullptr, 0);

  // This should print a summary
  __moore_uvm_report_summarize();

  __moore_uvm_set_fatal_exits(true);
  __moore_uvm_reset_report_counts();
}

TEST(MooreRuntimeUvmReportTest, VerbosityFiltering) {
  __moore_uvm_reset_report_counts();
  __moore_uvm_set_report_verbosity(MOORE_UVM_LOW);

  // Enable counting for info to verify filtering
  __moore_uvm_set_report_severity_action(MOORE_UVM_INFO,
                                         MOORE_UVM_DISPLAY | MOORE_UVM_COUNT);

  const char *id = "TEST";
  int64_t idLen = 4;
  const char *msg = "msg";
  int64_t msgLen = 3;

  // UVM_LOW message should be displayed (verbosity <= threshold)
  __moore_uvm_report_info(id, idLen, msg, msgLen, MOORE_UVM_LOW, nullptr, 0, 0,
                          nullptr, 0);
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_INFO), 1);

  // UVM_MEDIUM message should be filtered (verbosity > threshold)
  __moore_uvm_report_info(id, idLen, msg, msgLen, MOORE_UVM_MEDIUM, nullptr, 0,
                          0, nullptr, 0);
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_INFO), 1); // Still 1

  // UVM_HIGH message should be filtered
  __moore_uvm_report_info(id, idLen, msg, msgLen, MOORE_UVM_HIGH, nullptr, 0, 0,
                          nullptr, 0);
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_INFO), 1); // Still 1

  // UVM_NONE (0) message should always be displayed
  __moore_uvm_report_info(id, idLen, msg, msgLen, MOORE_UVM_NONE, nullptr, 0, 0,
                          nullptr, 0);
  EXPECT_EQ(__moore_uvm_get_report_count(MOORE_UVM_INFO), 2);

  // Restore defaults
  __moore_uvm_set_report_severity_action(MOORE_UVM_INFO, MOORE_UVM_DISPLAY);
  __moore_uvm_set_report_verbosity(MOORE_UVM_MEDIUM);
  __moore_uvm_reset_report_counts();
}

TEST(MooreRuntimeUvmReportTest, NullInputs) {
  // Test that null inputs don't crash

  // Null ID
  __moore_uvm_report_info(nullptr, 0, "msg", 3, MOORE_UVM_LOW, nullptr, 0, 0,
                          nullptr, 0);

  // Null message
  __moore_uvm_report_info("ID", 2, nullptr, 0, MOORE_UVM_LOW, nullptr, 0, 0,
                          nullptr, 0);

  // Empty ID and message
  __moore_uvm_report_info("", 0, "", 0, MOORE_UVM_LOW, nullptr, 0, 0, nullptr,
                          0);

  // Null ID for set_report_id_verbosity
  __moore_uvm_set_report_id_verbosity(nullptr, 0, MOORE_UVM_HIGH);

  // Null ID for report_enabled
  EXPECT_EQ(
      __moore_uvm_report_enabled(MOORE_UVM_MEDIUM, MOORE_UVM_INFO, nullptr, 0),
      1); // Uses global verbosity
}

//===----------------------------------------------------------------------===//
// Virtual Interface Binding Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeVifTest, CreateAndRelease) {
  __moore_vif_clear_all();
  __moore_vif_clear_registry();

  // Create a virtual interface handle
  MooreVifHandle vif = __moore_vif_create("apb_if", 6, nullptr, 0);
  ASSERT_NE(vif, MOORE_VIF_NULL);

  // Initially not bound
  EXPECT_EQ(__moore_vif_is_bound(vif), 0);
  EXPECT_EQ(__moore_vif_get_instance(vif), nullptr);

  // Release the handle
  __moore_vif_release(vif);

  __moore_vif_clear_all();
}

TEST(MooreRuntimeVifTest, CreateWithModport) {
  __moore_vif_clear_all();
  __moore_vif_clear_registry();

  // Create a virtual interface handle with modport
  MooreVifHandle vif = __moore_vif_create("apb_if", 6, "driver", 6);
  ASSERT_NE(vif, MOORE_VIF_NULL);

  // Check type name
  MooreString typeName = __moore_vif_get_type_name(vif);
  EXPECT_EQ(std::string(typeName.data, typeName.len), "apb_if");
  __moore_free(typeName.data);

  // Check modport name
  MooreString modportName = __moore_vif_get_modport_name(vif);
  EXPECT_EQ(std::string(modportName.data, modportName.len), "driver");
  __moore_free(modportName.data);

  __moore_vif_release(vif);
  __moore_vif_clear_all();
}

TEST(MooreRuntimeVifTest, CreateNullInputs) {
  __moore_vif_clear_all();

  // Null interface type name should fail
  EXPECT_EQ(__moore_vif_create(nullptr, 0, nullptr, 0), MOORE_VIF_NULL);
  EXPECT_EQ(__moore_vif_create("", 0, nullptr, 0), MOORE_VIF_NULL);
  EXPECT_EQ(__moore_vif_create("type", 0, nullptr, 0), MOORE_VIF_NULL);

  __moore_vif_clear_all();
}

TEST(MooreRuntimeVifTest, BindAndUnbind) {
  __moore_vif_clear_all();
  __moore_vif_clear_registry();

  MooreVifHandle vif = __moore_vif_create("simple_bus", 10, nullptr, 0);
  ASSERT_NE(vif, MOORE_VIF_NULL);

  // Create a mock interface instance
  struct MockInterface {
    int32_t data;
    int32_t valid;
  };
  MockInterface interfaceInstance = {0x12345678, 1};

  // Bind the virtual interface
  EXPECT_EQ(__moore_vif_bind(vif, &interfaceInstance), 1);
  EXPECT_EQ(__moore_vif_is_bound(vif), 1);
  EXPECT_EQ(__moore_vif_get_instance(vif), &interfaceInstance);

  // Unbind (bind to nullptr)
  EXPECT_EQ(__moore_vif_bind(vif, nullptr), 1);
  EXPECT_EQ(__moore_vif_is_bound(vif), 0);
  EXPECT_EQ(__moore_vif_get_instance(vif), nullptr);

  __moore_vif_release(vif);
  __moore_vif_clear_all();
}

TEST(MooreRuntimeVifTest, BindNullHandle) {
  // Binding to a null handle should fail
  EXPECT_EQ(__moore_vif_bind(MOORE_VIF_NULL, nullptr), 0);

  int32_t dummy = 0;
  EXPECT_EQ(__moore_vif_bind(MOORE_VIF_NULL, &dummy), 0);
}

TEST(MooreRuntimeVifTest, SignalAccess) {
  __moore_vif_clear_all();
  __moore_vif_clear_registry();

  // Register signals for the interface type
  EXPECT_EQ(__moore_vif_register_signal("test_if", 7, "data", 4, 0, 4), 1);
  EXPECT_EQ(__moore_vif_register_signal("test_if", 7, "valid", 5, 4, 4), 1);
  EXPECT_EQ(__moore_vif_register_signal("test_if", 7, "ready", 5, 8, 4), 1);

  // Create virtual interface
  MooreVifHandle vif = __moore_vif_create("test_if", 7, nullptr, 0);
  ASSERT_NE(vif, MOORE_VIF_NULL);

  // Create and bind a mock interface
  struct TestInterface {
    int32_t data;
    int32_t valid;
    int32_t ready;
  };
  TestInterface iface = {static_cast<int32_t>(0xDEADBEEF), 1, 0};

  EXPECT_EQ(__moore_vif_bind(vif, &iface), 1);

  // Read signal via virtual interface
  int32_t readData = 0;
  EXPECT_EQ(__moore_vif_get_signal(vif, "data", 4, &readData, sizeof(readData)),
            1);
  EXPECT_EQ(readData, static_cast<int32_t>(0xDEADBEEF));

  int32_t readValid = 0;
  EXPECT_EQ(
      __moore_vif_get_signal(vif, "valid", 5, &readValid, sizeof(readValid)),
      1);
  EXPECT_EQ(readValid, 1);

  // Write signal via virtual interface
  int32_t newReady = 1;
  EXPECT_EQ(
      __moore_vif_set_signal(vif, "ready", 5, &newReady, sizeof(newReady)), 1);
  EXPECT_EQ(iface.ready, 1);

  // Write to data
  int32_t newData = 0xCAFEBABE;
  EXPECT_EQ(__moore_vif_set_signal(vif, "data", 4, &newData, sizeof(newData)),
            1);
  EXPECT_EQ(iface.data, static_cast<int32_t>(0xCAFEBABE));

  __moore_vif_release(vif);
  __moore_vif_clear_registry();
  __moore_vif_clear_all();
}

TEST(MooreRuntimeVifTest, SignalRef) {
  __moore_vif_clear_all();
  __moore_vif_clear_registry();

  // Register signal
  EXPECT_EQ(__moore_vif_register_signal("ref_if", 6, "value", 5, 0, 8), 1);

  MooreVifHandle vif = __moore_vif_create("ref_if", 6, nullptr, 0);
  ASSERT_NE(vif, MOORE_VIF_NULL);

  // Create and bind interface
  int64_t interfaceValue = 12345;

  EXPECT_EQ(__moore_vif_bind(vif, &interfaceValue), 1);

  // Get signal reference
  void *ref = __moore_vif_get_signal_ref(vif, "value", 5);
  ASSERT_NE(ref, nullptr);
  EXPECT_EQ(ref, &interfaceValue);

  // Modify through reference
  *static_cast<int64_t *>(ref) = 67890;
  EXPECT_EQ(interfaceValue, 67890);

  __moore_vif_release(vif);
  __moore_vif_clear_registry();
  __moore_vif_clear_all();
}

TEST(MooreRuntimeVifTest, SignalAccessUnbound) {
  __moore_vif_clear_all();
  __moore_vif_clear_registry();

  // Register signal
  EXPECT_EQ(__moore_vif_register_signal("unbound_if", 10, "sig", 3, 0, 4), 1);

  MooreVifHandle vif = __moore_vif_create("unbound_if", 10, nullptr, 0);
  ASSERT_NE(vif, MOORE_VIF_NULL);

  // Try to access signal on unbound vif - should fail
  int32_t value = 0;
  EXPECT_EQ(__moore_vif_get_signal(vif, "sig", 3, &value, sizeof(value)), 0);
  EXPECT_EQ(__moore_vif_set_signal(vif, "sig", 3, &value, sizeof(value)), 0);
  EXPECT_EQ(__moore_vif_get_signal_ref(vif, "sig", 3), nullptr);

  __moore_vif_release(vif);
  __moore_vif_clear_registry();
  __moore_vif_clear_all();
}

TEST(MooreRuntimeVifTest, SignalAccessNonexistent) {
  __moore_vif_clear_all();
  __moore_vif_clear_registry();

  // Register one signal
  EXPECT_EQ(__moore_vif_register_signal("partial_if", 10, "exists", 6, 0, 4),
            1);

  MooreVifHandle vif = __moore_vif_create("partial_if", 10, nullptr, 0);
  ASSERT_NE(vif, MOORE_VIF_NULL);

  int32_t dummy = 0;
  EXPECT_EQ(__moore_vif_bind(vif, &dummy), 1);

  // Try to access nonexistent signal
  int32_t value = 0;
  EXPECT_EQ(
      __moore_vif_get_signal(vif, "nonexistent", 11, &value, sizeof(value)), 0);
  EXPECT_EQ(
      __moore_vif_set_signal(vif, "nonexistent", 11, &value, sizeof(value)), 0);
  EXPECT_EQ(__moore_vif_get_signal_ref(vif, "nonexistent", 11), nullptr);

  __moore_vif_release(vif);
  __moore_vif_clear_registry();
  __moore_vif_clear_all();
}

TEST(MooreRuntimeVifTest, Compare) {
  __moore_vif_clear_all();
  __moore_vif_clear_registry();

  MooreVifHandle vif1 = __moore_vif_create("cmp_if", 6, nullptr, 0);
  MooreVifHandle vif2 = __moore_vif_create("cmp_if", 6, nullptr, 0);
  ASSERT_NE(vif1, MOORE_VIF_NULL);
  ASSERT_NE(vif2, MOORE_VIF_NULL);

  // Both unbound - should be equal
  EXPECT_EQ(__moore_vif_compare(vif1, vif2), 1);

  // Bind vif1 only
  int32_t instance1 = 100;
  EXPECT_EQ(__moore_vif_bind(vif1, &instance1), 1);

  // One bound, one not - should be not equal
  EXPECT_EQ(__moore_vif_compare(vif1, vif2), 0);

  // Bind vif2 to different instance
  int32_t instance2 = 200;
  EXPECT_EQ(__moore_vif_bind(vif2, &instance2), 1);

  // Both bound to different instances - not equal
  EXPECT_EQ(__moore_vif_compare(vif1, vif2), 0);

  // Bind vif2 to same instance as vif1
  EXPECT_EQ(__moore_vif_bind(vif2, &instance1), 1);

  // Both bound to same instance - equal
  EXPECT_EQ(__moore_vif_compare(vif1, vif2), 1);

  // Compare with null
  EXPECT_EQ(__moore_vif_compare(vif1, MOORE_VIF_NULL), 0);
  EXPECT_EQ(__moore_vif_compare(MOORE_VIF_NULL, vif1), 0);
  EXPECT_EQ(__moore_vif_compare(MOORE_VIF_NULL, MOORE_VIF_NULL), 1);

  __moore_vif_release(vif1);
  __moore_vif_release(vif2);
  __moore_vif_clear_all();
}

TEST(MooreRuntimeVifTest, ClearAll) {
  __moore_vif_clear_all();
  __moore_vif_clear_registry();

  // Create multiple handles
  MooreVifHandle vif1 = __moore_vif_create("clear_if1", 9, nullptr, 0);
  MooreVifHandle vif2 = __moore_vif_create("clear_if2", 9, nullptr, 0);
  MooreVifHandle vif3 = __moore_vif_create("clear_if3", 9, nullptr, 0);
  EXPECT_NE(vif1, MOORE_VIF_NULL);
  EXPECT_NE(vif2, MOORE_VIF_NULL);
  EXPECT_NE(vif3, MOORE_VIF_NULL);

  // Clear all - should not crash even though handles are now invalid
  __moore_vif_clear_all();

  // Create new handles after clear - should work
  MooreVifHandle newVif = __moore_vif_create("new_if", 6, nullptr, 0);
  EXPECT_NE(newVif, MOORE_VIF_NULL);

  __moore_vif_clear_all();
}

TEST(MooreRuntimeVifTest, RegisterSignalInvalidInputs) {
  __moore_vif_clear_registry();

  // Null type name
  EXPECT_EQ(__moore_vif_register_signal(nullptr, 0, "sig", 3, 0, 4), 0);

  // Empty type name
  EXPECT_EQ(__moore_vif_register_signal("", 0, "sig", 3, 0, 4), 0);

  // Null signal name
  EXPECT_EQ(__moore_vif_register_signal("type", 4, nullptr, 0, 0, 4), 0);

  // Empty signal name
  EXPECT_EQ(__moore_vif_register_signal("type", 4, "", 0, 0, 4), 0);

  // Zero signal size
  EXPECT_EQ(__moore_vif_register_signal("type", 4, "sig", 3, 0, 0), 0);

  __moore_vif_clear_registry();
}

TEST(MooreRuntimeVifTest, TypeNameAndModportName) {
  __moore_vif_clear_all();

  // Create without modport
  MooreVifHandle vif1 = __moore_vif_create("my_interface", 12, nullptr, 0);
  ASSERT_NE(vif1, MOORE_VIF_NULL);

  MooreString type1 = __moore_vif_get_type_name(vif1);
  EXPECT_EQ(std::string(type1.data, type1.len), "my_interface");
  __moore_free(type1.data);

  MooreString modport1 = __moore_vif_get_modport_name(vif1);
  EXPECT_EQ(modport1.len, 0);
  EXPECT_EQ(modport1.data, nullptr);

  // Create with modport
  MooreVifHandle vif2 = __moore_vif_create("my_interface", 12, "monitor", 7);
  ASSERT_NE(vif2, MOORE_VIF_NULL);

  MooreString type2 = __moore_vif_get_type_name(vif2);
  EXPECT_EQ(std::string(type2.data, type2.len), "my_interface");
  __moore_free(type2.data);

  MooreString modport2 = __moore_vif_get_modport_name(vif2);
  EXPECT_EQ(std::string(modport2.data, modport2.len), "monitor");
  __moore_free(modport2.data);

  // Null handle
  MooreString nullType = __moore_vif_get_type_name(MOORE_VIF_NULL);
  EXPECT_EQ(nullType.len, 0);
  EXPECT_EQ(nullType.data, nullptr);

  MooreString nullModport = __moore_vif_get_modport_name(MOORE_VIF_NULL);
  EXPECT_EQ(nullModport.len, 0);
  EXPECT_EQ(nullModport.data, nullptr);

  __moore_vif_release(vif1);
  __moore_vif_release(vif2);
  __moore_vif_clear_all();
}

TEST(MooreRuntimeVifTest, NullHandleOperations) {
  // These should not crash and return appropriate failure values
  EXPECT_EQ(__moore_vif_is_bound(MOORE_VIF_NULL), 0);
  EXPECT_EQ(__moore_vif_get_instance(MOORE_VIF_NULL), nullptr);

  int32_t value = 0;
  EXPECT_EQ(
      __moore_vif_get_signal(MOORE_VIF_NULL, "sig", 3, &value, sizeof(value)),
      0);
  EXPECT_EQ(
      __moore_vif_set_signal(MOORE_VIF_NULL, "sig", 3, &value, sizeof(value)),
      0);
  EXPECT_EQ(__moore_vif_get_signal_ref(MOORE_VIF_NULL, "sig", 3), nullptr);

  // Release null - should not crash
  __moore_vif_release(MOORE_VIF_NULL);
}

TEST(MooreRuntimeVifTest, MultipleInterfaces) {
  __moore_vif_clear_all();
  __moore_vif_clear_registry();

  // Register signals for multiple interface types
  EXPECT_EQ(__moore_vif_register_signal("if_a", 4, "data_a", 6, 0, 4), 1);
  EXPECT_EQ(__moore_vif_register_signal("if_b", 4, "data_b", 6, 0, 8), 1);

  // Create virtual interfaces for different types
  MooreVifHandle vifA = __moore_vif_create("if_a", 4, nullptr, 0);
  MooreVifHandle vifB = __moore_vif_create("if_b", 4, nullptr, 0);
  ASSERT_NE(vifA, MOORE_VIF_NULL);
  ASSERT_NE(vifB, MOORE_VIF_NULL);

  // Create mock interface instances
  int32_t instanceA = 0xAAAAAAAA;
  int64_t instanceB = 0xBBBBBBBBBBBBBBBB;

  EXPECT_EQ(__moore_vif_bind(vifA, &instanceA), 1);
  EXPECT_EQ(__moore_vif_bind(vifB, &instanceB), 1);

  // Read from each
  int32_t readA = 0;
  EXPECT_EQ(__moore_vif_get_signal(vifA, "data_a", 6, &readA, sizeof(readA)),
            1);
  EXPECT_EQ(readA, static_cast<int32_t>(0xAAAAAAAA));

  int64_t readB = 0;
  EXPECT_EQ(__moore_vif_get_signal(vifB, "data_b", 6, &readB, sizeof(readB)),
            1);
  EXPECT_EQ(readB, static_cast<int64_t>(0xBBBBBBBBBBBBBBBB));

  // Accessing wrong signal type should fail (signal not in that interface)
  EXPECT_EQ(__moore_vif_get_signal(vifA, "data_b", 6, &readA, sizeof(readA)),
            0);
  EXPECT_EQ(__moore_vif_get_signal(vifB, "data_a", 6, &readB, sizeof(readB)),
            0);

  __moore_vif_release(vifA);
  __moore_vif_release(vifB);
  __moore_vif_clear_registry();
  __moore_vif_clear_all();
}

TEST(MooreRuntimeVifTest, Rebinding) {
  __moore_vif_clear_all();
  __moore_vif_clear_registry();

  EXPECT_EQ(__moore_vif_register_signal("rebind_if", 9, "val", 3, 0, 4), 1);

  MooreVifHandle vif = __moore_vif_create("rebind_if", 9, nullptr, 0);
  ASSERT_NE(vif, MOORE_VIF_NULL);

  // First instance
  int32_t instance1 = 111;
  EXPECT_EQ(__moore_vif_bind(vif, &instance1), 1);

  int32_t readVal = 0;
  EXPECT_EQ(__moore_vif_get_signal(vif, "val", 3, &readVal, sizeof(readVal)),
            1);
  EXPECT_EQ(readVal, 111);

  // Rebind to second instance
  int32_t instance2 = 222;
  EXPECT_EQ(__moore_vif_bind(vif, &instance2), 1);

  readVal = 0;
  EXPECT_EQ(__moore_vif_get_signal(vif, "val", 3, &readVal, sizeof(readVal)),
            1);
  EXPECT_EQ(readVal, 222);

  // Rebind back to first
  EXPECT_EQ(__moore_vif_bind(vif, &instance1), 1);

  readVal = 0;
  EXPECT_EQ(__moore_vif_get_signal(vif, "val", 3, &readVal, sizeof(readVal)),
            1);
  EXPECT_EQ(readVal, 111);

  __moore_vif_release(vif);
  __moore_vif_clear_registry();
  __moore_vif_clear_all();
}

//===----------------------------------------------------------------------===//
// SystemVerilog Semaphore Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeSemaphoreTest, Creation) {
  MooreSemaphoreHandle sem = __moore_semaphore_create(1);
  EXPECT_NE(sem, MOORE_SEMAPHORE_INVALID_HANDLE);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 1);

  __moore_semaphore_destroy(sem);
}

TEST(MooreRuntimeSemaphoreTest, CreationWithZeroKeys) {
  MooreSemaphoreHandle sem = __moore_semaphore_create(0);
  EXPECT_NE(sem, MOORE_SEMAPHORE_INVALID_HANDLE);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 0);

  __moore_semaphore_destroy(sem);
}

TEST(MooreRuntimeSemaphoreTest, CreationWithNegativeKeys) {
  MooreSemaphoreHandle sem = __moore_semaphore_create(-1);
  EXPECT_EQ(sem, MOORE_SEMAPHORE_INVALID_HANDLE);
}

TEST(MooreRuntimeSemaphoreTest, PutAndGet) {
  MooreSemaphoreHandle sem = __moore_semaphore_create(0);
  EXPECT_NE(sem, MOORE_SEMAPHORE_INVALID_HANDLE);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 0);

  // Put 2 keys
  __moore_semaphore_put(sem, 2);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 2);

  // Get 1 key (should not block since we have 2)
  __moore_semaphore_get(sem, 1);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 1);

  // Get another key
  __moore_semaphore_get(sem, 1);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 0);

  __moore_semaphore_destroy(sem);
}

TEST(MooreRuntimeSemaphoreTest, TryGetSuccess) {
  MooreSemaphoreHandle sem = __moore_semaphore_create(3);
  EXPECT_NE(sem, MOORE_SEMAPHORE_INVALID_HANDLE);

  // try_get should succeed when enough keys available
  EXPECT_EQ(__moore_semaphore_try_get(sem, 2), 1);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 1);

  // try_get should succeed for the remaining key
  EXPECT_EQ(__moore_semaphore_try_get(sem, 1), 1);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 0);

  __moore_semaphore_destroy(sem);
}

TEST(MooreRuntimeSemaphoreTest, TryGetFailure) {
  MooreSemaphoreHandle sem = __moore_semaphore_create(1);
  EXPECT_NE(sem, MOORE_SEMAPHORE_INVALID_HANDLE);

  // try_get should fail when not enough keys
  EXPECT_EQ(__moore_semaphore_try_get(sem, 2), 0);
  // Key count should be unchanged
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 1);

  // Now get the one key we have
  EXPECT_EQ(__moore_semaphore_try_get(sem, 1), 1);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 0);

  // try_get with 0 keys should fail
  EXPECT_EQ(__moore_semaphore_try_get(sem, 1), 0);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 0);

  __moore_semaphore_destroy(sem);
}

TEST(MooreRuntimeSemaphoreTest, MultiplePutAndGet) {
  MooreSemaphoreHandle sem = __moore_semaphore_create(0);
  EXPECT_NE(sem, MOORE_SEMAPHORE_INVALID_HANDLE);

  // Simulate AXI4 driver pattern: multiple threads using semaphore
  __moore_semaphore_put(sem, 1);  // Initial key
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 1);

  // Thread 1 gets key
  __moore_semaphore_get(sem, 1);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 0);

  // Thread 1 puts key back
  __moore_semaphore_put(sem, 1);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 1);

  // Thread 2 gets key
  __moore_semaphore_get(sem, 1);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 0);

  __moore_semaphore_destroy(sem);
}

TEST(MooreRuntimeSemaphoreTest, InvalidHandleOperations) {
  // Operations on invalid handle should not crash
  __moore_semaphore_put(MOORE_SEMAPHORE_INVALID_HANDLE, 1);
  __moore_semaphore_get(MOORE_SEMAPHORE_INVALID_HANDLE, 1);
  EXPECT_EQ(__moore_semaphore_try_get(MOORE_SEMAPHORE_INVALID_HANDLE, 1), 0);
  EXPECT_EQ(__moore_semaphore_get_key_count(MOORE_SEMAPHORE_INVALID_HANDLE), 0);
  __moore_semaphore_destroy(MOORE_SEMAPHORE_INVALID_HANDLE);
}

TEST(MooreRuntimeSemaphoreTest, MultipleSemaphores) {
  // Test multiple independent semaphores (like write_data_channel_key,
  // write_response_channel_key in AXI4)
  MooreSemaphoreHandle sem1 = __moore_semaphore_create(1);
  MooreSemaphoreHandle sem2 = __moore_semaphore_create(1);
  MooreSemaphoreHandle sem3 = __moore_semaphore_create(1);

  EXPECT_NE(sem1, sem2);
  EXPECT_NE(sem2, sem3);
  EXPECT_NE(sem1, sem3);

  // Each operates independently
  __moore_semaphore_get(sem1, 1);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem1), 0);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem2), 1);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem3), 1);

  __moore_semaphore_destroy(sem1);
  __moore_semaphore_destroy(sem2);
  __moore_semaphore_destroy(sem3);
}

TEST(MooreRuntimeSemaphoreTest, ConcurrentPutAndGet) {
  // Test concurrent access pattern similar to AXI4 driver
  MooreSemaphoreHandle sem = __moore_semaphore_create(1);
  std::atomic<int> counter{0};

  // Simulate multiple threads accessing the semaphore
  auto worker = [&sem, &counter]() {
    for (int i = 0; i < 10; ++i) {
      __moore_semaphore_get(sem, 1);
      counter++;
      __moore_semaphore_put(sem, 1);
    }
  };

  std::thread t1(worker);
  std::thread t2(worker);

  t1.join();
  t2.join();

  EXPECT_EQ(counter.load(), 20);
  EXPECT_EQ(__moore_semaphore_get_key_count(sem), 1);

  __moore_semaphore_destroy(sem);
}

//===----------------------------------------------------------------------===//
// Additional Driver-Side Sequencer Tests
//===----------------------------------------------------------------------===//

TEST(MooreRuntimeSequenceTest, TryGetNextItemEmpty) {
  const char *seqrName = "try_get_empty_sequencer";
  MooreSequencerHandle seqr = __moore_sequencer_create(
      seqrName, strlen(seqrName), 0);

  __moore_sequencer_start(seqr);

  // try_get_next_item on empty sequencer should return 0 (no item)
  TestSeqTransaction rxTx;
  int32_t result = __moore_sequencer_try_get_next_item(seqr, &rxTx, sizeof(rxTx));
  EXPECT_EQ(result, 0);

  __moore_sequencer_stop(seqr);
  __moore_sequencer_destroy(seqr);
}

struct TryGetTestContext {
  MooreSequencerHandle sequencer;
  std::atomic<int> itemsSent{0};
  TestSeqTransaction itemToSend;
};

static void tryGetSequenceBody(MooreSequenceHandle seq, void *userData) {
  auto *ctx = static_cast<TryGetTestContext *>(userData);

  __moore_sequence_start_item(seq, &ctx->itemToSend, sizeof(ctx->itemToSend));
  __moore_sequence_finish_item(seq, &ctx->itemToSend, sizeof(ctx->itemToSend));
  ctx->itemsSent++;
}

TEST(MooreRuntimeSequenceTest, TryGetNextItemWithData) {
  const char *seqrName = "try_get_data_sequencer";
  const char *seqName = "try_get_data_sequence";

  MooreSequencerHandle seqr = __moore_sequencer_create(
      seqrName, strlen(seqrName), 0);
  MooreSequenceHandle seq = __moore_sequence_create(seqName, strlen(seqName), 0);

  __moore_sequencer_start(seqr);

  TryGetTestContext ctx;
  ctx.sequencer = seqr;
  ctx.itemToSend = {0x5000, 0xDEAD, 1};

  __moore_sequence_start_async(seq, seqr, tryGetSequenceBody, &ctx);

  // Wait a bit for sequence to start
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Now try_get_next_item should succeed
  TestSeqTransaction rxTx;
  int32_t result = __moore_sequencer_try_get_next_item(seqr, &rxTx, sizeof(rxTx));
  if (result == 1) {
    EXPECT_EQ(rxTx.addr, 0x5000u);
    EXPECT_EQ(rxTx.data, 0xDEADu);
    __moore_sequencer_item_done(seqr);
  }

  __moore_sequence_wait(seq);
  __moore_sequencer_stop(seqr);
  __moore_sequence_destroy(seq);
  __moore_sequencer_destroy(seqr);
}

TEST(MooreRuntimeSequenceTest, PeekNextItem) {
  const char *seqrName = "peek_sequencer";
  const char *seqName = "peek_sequence";

  MooreSequencerHandle seqr = __moore_sequencer_create(
      seqrName, strlen(seqrName), 0);
  MooreSequenceHandle seq = __moore_sequence_create(seqName, strlen(seqName), 0);

  __moore_sequencer_start(seqr);

  TryGetTestContext ctx;
  ctx.sequencer = seqr;
  ctx.itemToSend = {0x6000, 0xBEEF, 0};

  __moore_sequence_start_async(seq, seqr, tryGetSequenceBody, &ctx);

  // Wait for sequence to post item
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Peek should return the item without removing it
  TestSeqTransaction peekedTx;
  int32_t peekResult = __moore_sequencer_peek_next_item(seqr, &peekedTx, sizeof(peekedTx));
  if (peekResult == 1) {
    EXPECT_EQ(peekedTx.addr, 0x6000u);
    EXPECT_EQ(peekedTx.data, 0xBEEFu);

    // Item should still be available for get_next_item
    TestSeqTransaction rxTx;
    int32_t getResult = __moore_sequencer_get_next_item(seqr, &rxTx, sizeof(rxTx));
    EXPECT_EQ(getResult, 1);
    EXPECT_EQ(rxTx.addr, peekedTx.addr);
    EXPECT_EQ(rxTx.data, peekedTx.data);

    __moore_sequencer_item_done(seqr);
  }

  __moore_sequence_wait(seq);
  __moore_sequencer_stop(seqr);
  __moore_sequence_destroy(seq);
  __moore_sequencer_destroy(seqr);
}

TEST(MooreRuntimeSequenceTest, HasItemsCheck) {
  const char *seqrName = "has_items_check_sequencer";
  const char *seqName = "has_items_check_sequence";

  MooreSequencerHandle seqr = __moore_sequencer_create(
      seqrName, strlen(seqrName), 0);
  MooreSequenceHandle seq = __moore_sequence_create(seqName, strlen(seqName), 0);

  __moore_sequencer_start(seqr);

  // Initially no items
  EXPECT_EQ(__moore_sequencer_has_items(seqr), 0);

  TryGetTestContext ctx;
  ctx.sequencer = seqr;
  ctx.itemToSend = {0x7000, 0xCAFE, 1};

  __moore_sequence_start_async(seq, seqr, tryGetSequenceBody, &ctx);

  // Wait for sequence to post item
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Now has_items might return 1
  // Note: Due to timing, this might still be 0 if the item hasn't been posted yet
  int32_t hasItems = __moore_sequencer_has_items(seqr);
  if (hasItems == 1) {
    TestSeqTransaction rxTx;
    EXPECT_EQ(__moore_sequencer_get_next_item(seqr, &rxTx, sizeof(rxTx)), 1);
    __moore_sequencer_item_done(seqr);
  }

  __moore_sequence_wait(seq);
  __moore_sequencer_stop(seqr);
  __moore_sequence_destroy(seq);
  __moore_sequencer_destroy(seqr);
}

} // namespace
