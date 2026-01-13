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

} // namespace
