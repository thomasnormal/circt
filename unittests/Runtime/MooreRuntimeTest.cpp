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
#include <cstddef>
#include <cstring>

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

} // namespace
