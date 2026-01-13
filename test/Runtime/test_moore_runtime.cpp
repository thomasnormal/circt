//===- test_moore_runtime.cpp - Test Moore runtime library ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple test for the Moore runtime library string operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Runtime/MooreRuntime.h"
#include <cassert>
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
  std::cout << "Testing Moore Runtime Library\n";
  std::cout << "==============================\n\n";

  // Test 1: String length
  {
    char data[] = "hello";
    MooreString str = {data, 5};
    CHECK(__moore_string_len(&str) == 5, "string length should be 5");

    MooreString empty = {nullptr, 0};
    CHECK(__moore_string_len(&empty) == 0, "empty string length should be 0");
    CHECK(__moore_string_len(nullptr) == 0, "null string length should be 0");
    PASS("String length tests");
  }

  // Test 2: String toupper
  {
    char data[] = "Hello World";
    MooreString str = {data, 11};
    MooreString result = __moore_string_toupper(&str);

    CHECK(result.data != nullptr, "toupper result should not be null");
    CHECK(result.len == 11, "toupper result length should be 11");
    CHECK(std::string(result.data, result.len) == "HELLO WORLD",
          "toupper should convert to uppercase");

    __moore_free(result.data);
    PASS("String toupper tests");
  }

  // Test 3: String tolower
  {
    char data[] = "Hello World";
    MooreString str = {data, 11};
    MooreString result = __moore_string_tolower(&str);

    CHECK(result.data != nullptr, "tolower result should not be null");
    CHECK(result.len == 11, "tolower result length should be 11");
    CHECK(std::string(result.data, result.len) == "hello world",
          "tolower should convert to lowercase");

    __moore_free(result.data);
    PASS("String tolower tests");
  }

  // Test 4: String getc
  {
    char data[] = "abc";
    MooreString str = {data, 3};

    CHECK(__moore_string_getc(&str, 0) == 'a', "getc(0) should return 'a'");
    CHECK(__moore_string_getc(&str, 1) == 'b', "getc(1) should return 'b'");
    CHECK(__moore_string_getc(&str, 2) == 'c', "getc(2) should return 'c'");
    CHECK(__moore_string_getc(&str, 3) == 0, "getc(3) out of bounds should return 0");
    CHECK(__moore_string_getc(&str, -1) == 0, "getc(-1) should return 0");
    PASS("String getc tests");
  }

  // Test 5: String substr
  {
    char data[] = "hello world";
    MooreString str = {data, 11};

    MooreString sub = __moore_string_substr(&str, 0, 5);
    CHECK(sub.data != nullptr, "substr result should not be null");
    CHECK(sub.len == 5, "substr length should be 5");
    CHECK(std::string(sub.data, sub.len) == "hello", "substr(0,5) should be 'hello'");
    __moore_free(sub.data);

    sub = __moore_string_substr(&str, 6, 5);
    CHECK(std::string(sub.data, sub.len) == "world", "substr(6,5) should be 'world'");
    __moore_free(sub.data);

    // Clamped to bounds
    sub = __moore_string_substr(&str, 8, 10);
    CHECK(sub.len == 3, "substr clamped length should be 3");
    CHECK(std::string(sub.data, sub.len) == "rld", "substr(8,10) should be 'rld'");
    __moore_free(sub.data);

    PASS("String substr tests");
  }

  // Test 6: String itoa
  {
    MooreString result = __moore_string_itoa(42);
    CHECK(result.data != nullptr, "itoa result should not be null");
    CHECK(std::string(result.data, result.len) == "42", "itoa(42) should be '42'");
    __moore_free(result.data);

    result = __moore_string_itoa(-123);
    CHECK(std::string(result.data, result.len) == "-123", "itoa(-123) should be '-123'");
    __moore_free(result.data);

    result = __moore_string_itoa(0);
    CHECK(std::string(result.data, result.len) == "0", "itoa(0) should be '0'");
    __moore_free(result.data);

    PASS("String itoa tests");
  }

  // Test 7: String concat
  {
    char data1[] = "hello";
    char data2[] = " world";
    MooreString lhs = {data1, 5};
    MooreString rhs = {data2, 6};

    MooreString result = __moore_string_concat(&lhs, &rhs);
    CHECK(result.data != nullptr, "concat result should not be null");
    CHECK(result.len == 11, "concat length should be 11");
    CHECK(std::string(result.data, result.len) == "hello world",
          "concat should produce 'hello world'");
    __moore_free(result.data);

    // Concat with empty
    MooreString empty = {nullptr, 0};
    result = __moore_string_concat(&lhs, &empty);
    CHECK(std::string(result.data, result.len) == "hello",
          "concat with empty should produce 'hello'");
    __moore_free(result.data);

    PASS("String concat tests");
  }

  // Test 8: String cmp
  {
    char data1[] = "abc";
    char data2[] = "abd";
    char data3[] = "abc";
    char data4[] = "ab";
    MooreString str1 = {data1, 3};
    MooreString str2 = {data2, 3};
    MooreString str3 = {data3, 3};
    MooreString str4 = {data4, 2};

    CHECK(__moore_string_cmp(&str1, &str3) == 0, "cmp('abc', 'abc') should be 0");
    CHECK(__moore_string_cmp(&str1, &str2) < 0, "cmp('abc', 'abd') should be < 0");
    CHECK(__moore_string_cmp(&str2, &str1) > 0, "cmp('abd', 'abc') should be > 0");
    CHECK(__moore_string_cmp(&str1, &str4) > 0, "cmp('abc', 'ab') should be > 0");
    CHECK(__moore_string_cmp(&str4, &str1) < 0, "cmp('ab', 'abc') should be < 0");

    PASS("String cmp tests");
  }

  // Test 9: String to int
  {
    char data1[] = "42";
    char data2[] = "-123";
    char data3[] = "0";
    MooreString str1 = {data1, 2};
    MooreString str2 = {data2, 4};
    MooreString str3 = {data3, 1};

    CHECK(__moore_string_to_int(&str1) == 42, "string_to_int('42') should be 42");
    CHECK(__moore_string_to_int(&str2) == -123, "string_to_int('-123') should be -123");
    CHECK(__moore_string_to_int(&str3) == 0, "string_to_int('0') should be 0");
    CHECK(__moore_string_to_int(nullptr) == 0, "string_to_int(null) should be 0");

    PASS("String to int tests");
  }

  // Test 10: Dynamic array
  {
    MooreQueue arr = __moore_dyn_array_new(10);
    CHECK(arr.data != nullptr, "dyn_array_new should return non-null");
    CHECK(arr.len == 10, "dyn_array_new length should be 10");

    // Check zeroed
    for (int i = 0; i < 10; ++i) {
      CHECK(static_cast<char*>(arr.data)[i] == 0, "array elements should be zeroed");
    }
    __moore_free(arr.data);

    // Test copy
    char source[] = "hello";
    arr = __moore_dyn_array_new_copy(5, source);
    CHECK(arr.data != nullptr, "dyn_array_new_copy should return non-null");
    CHECK(std::string(static_cast<char*>(arr.data), 5) == "hello",
          "dyn_array_new_copy should copy data");
    __moore_free(arr.data);

    // Test zero/negative size
    arr = __moore_dyn_array_new(0);
    CHECK(arr.data == nullptr, "dyn_array_new(0) should return null data");
    CHECK(arr.len == 0, "dyn_array_new(0) should have len 0");

    PASS("Dynamic array tests");
  }

  std::cout << "\n==============================\n";
  std::cout << "All tests passed!\n";
  return 0;
}
