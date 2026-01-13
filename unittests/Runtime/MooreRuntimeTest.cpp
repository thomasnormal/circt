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

namespace {

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

} // namespace
