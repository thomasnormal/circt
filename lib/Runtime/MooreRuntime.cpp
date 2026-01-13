//===- MooreRuntime.cpp - Runtime library for Moore dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the runtime library functions required by the
// MooreToCore lowering pass. These functions are linked with the compiled
// simulation binary to provide support for operations that cannot be lowered
// directly to LLVM IR.
//
// Priority focus: Queue and String operations (most commonly used in UVM).
//
//===----------------------------------------------------------------------===//

#include "circt/Runtime/MooreRuntime.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

//===----------------------------------------------------------------------===//
// Internal Helpers
//===----------------------------------------------------------------------===//

namespace {

/// Allocate a new string with the given length.
/// The caller is responsible for initializing the contents.
MooreString allocateString(int64_t len) {
  MooreString result;
  if (len > 0) {
    result.data = static_cast<char *>(std::malloc(len));
    result.len = len;
  } else {
    result.data = nullptr;
    result.len = 0;
  }
  return result;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Queue Operations
//===----------------------------------------------------------------------===//

extern "C" MooreQueue __moore_queue_max(MooreQueue *queue) {
  // TODO: Implement queue max operation.
  // This requires knowing the element type and size to compare elements.
  // For now, return an empty queue as a placeholder.
  MooreQueue result = {nullptr, 0};
  (void)queue; // Suppress unused parameter warning
  return result;
}

extern "C" MooreQueue __moore_queue_min(MooreQueue *queue) {
  // TODO: Implement queue min operation.
  // This requires knowing the element type and size to compare elements.
  // For now, return an empty queue as a placeholder.
  MooreQueue result = {nullptr, 0};
  (void)queue; // Suppress unused parameter warning
  return result;
}

//===----------------------------------------------------------------------===//
// Dynamic Array Operations
//===----------------------------------------------------------------------===//

extern "C" MooreQueue __moore_dyn_array_new(int32_t size) {
  MooreQueue result;
  if (size > 0) {
    // Allocate zeroed memory for the array.
    // Note: We allocate size bytes here; the caller must handle element sizing.
    result.data = std::calloc(size, 1);
    result.len = size;
  } else {
    result.data = nullptr;
    result.len = 0;
  }
  return result;
}

extern "C" MooreQueue __moore_dyn_array_new_copy(int32_t size, void *init) {
  MooreQueue result = __moore_dyn_array_new(size);
  if (result.data && init && size > 0) {
    std::memcpy(result.data, init, size);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Associative Array Operations
//===----------------------------------------------------------------------===//

extern "C" void __moore_assoc_delete(void *array) {
  // TODO: Implement associative array deletion.
  // This requires a proper associative array implementation.
  (void)array; // Suppress unused parameter warning
}

extern "C" void __moore_assoc_delete_key(void *array, void *key) {
  // TODO: Implement key deletion from associative array.
  (void)array;
  (void)key;
}

extern "C" bool __moore_assoc_first(void *array, void *key_out) {
  // TODO: Implement first key retrieval.
  (void)array;
  (void)key_out;
  return false;
}

extern "C" bool __moore_assoc_next(void *array, void *key_ref) {
  // TODO: Implement next key iteration.
  (void)array;
  (void)key_ref;
  return false;
}

extern "C" bool __moore_assoc_last(void *array, void *key_out) {
  // TODO: Implement last key retrieval.
  (void)array;
  (void)key_out;
  return false;
}

extern "C" bool __moore_assoc_prev(void *array, void *key_ref) {
  // TODO: Implement previous key iteration.
  (void)array;
  (void)key_ref;
  return false;
}

//===----------------------------------------------------------------------===//
// String Operations
//===----------------------------------------------------------------------===//

extern "C" int32_t __moore_string_len(MooreString *str) {
  if (!str || !str->data)
    return 0;
  return static_cast<int32_t>(str->len);
}

extern "C" MooreString __moore_string_toupper(MooreString *str) {
  if (!str || str->len <= 0 || !str->data) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  MooreString result = allocateString(str->len);
  for (int64_t i = 0; i < str->len; ++i) {
    result.data[i] = static_cast<char>(std::toupper(
        static_cast<unsigned char>(str->data[i])));
  }
  return result;
}

extern "C" MooreString __moore_string_tolower(MooreString *str) {
  if (!str || str->len <= 0 || !str->data) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  MooreString result = allocateString(str->len);
  for (int64_t i = 0; i < str->len; ++i) {
    result.data[i] = static_cast<char>(std::tolower(
        static_cast<unsigned char>(str->data[i])));
  }
  return result;
}

extern "C" int8_t __moore_string_getc(MooreString *str, int32_t index) {
  if (!str || !str->data || index < 0 || index >= str->len)
    return 0;
  return static_cast<int8_t>(str->data[index]);
}

extern "C" MooreString __moore_string_substr(MooreString *str, int32_t start,
                                              int32_t len) {
  if (!str || !str->data || start < 0 || len <= 0) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  // Clamp to string bounds
  if (start >= str->len) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  int64_t actualLen = std::min(static_cast<int64_t>(len), str->len - start);
  MooreString result = allocateString(actualLen);
  std::memcpy(result.data, str->data + start, actualLen);
  return result;
}

extern "C" MooreString __moore_string_itoa(int64_t value) {
  // Convert integer to decimal string
  char buffer[32]; // Enough for int64_t
  int len = std::snprintf(buffer, sizeof(buffer), "%ld", value);

  if (len <= 0) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  MooreString result = allocateString(len);
  std::memcpy(result.data, buffer, len);
  return result;
}

extern "C" MooreString __moore_string_concat(MooreString *lhs,
                                              MooreString *rhs) {
  int64_t lhsLen = (lhs && lhs->data) ? lhs->len : 0;
  int64_t rhsLen = (rhs && rhs->data) ? rhs->len : 0;

  if (lhsLen == 0 && rhsLen == 0) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  MooreString result = allocateString(lhsLen + rhsLen);
  if (lhsLen > 0)
    std::memcpy(result.data, lhs->data, lhsLen);
  if (rhsLen > 0)
    std::memcpy(result.data + lhsLen, rhs->data, rhsLen);

  return result;
}

extern "C" int32_t __moore_string_cmp(MooreString *lhs, MooreString *rhs) {
  // Handle null/empty cases
  bool lhsEmpty = !lhs || !lhs->data || lhs->len <= 0;
  bool rhsEmpty = !rhs || !rhs->data || rhs->len <= 0;

  if (lhsEmpty && rhsEmpty)
    return 0;
  if (lhsEmpty)
    return -1;
  if (rhsEmpty)
    return 1;

  // Compare up to the minimum length
  int64_t minLen = std::min(lhs->len, rhs->len);
  int cmp = std::memcmp(lhs->data, rhs->data, minLen);

  if (cmp != 0)
    return cmp;

  // If equal up to minLen, the shorter string is "less"
  if (lhs->len < rhs->len)
    return -1;
  if (lhs->len > rhs->len)
    return 1;
  return 0;
}

extern "C" MooreString __moore_int_to_string(int64_t value) {
  // Same implementation as itoa for unsigned interpretation in UVM context
  // For proper unsigned handling, we use %lu if the value should be unsigned
  char buffer[32];
  int len = std::snprintf(buffer, sizeof(buffer), "%lu",
                          static_cast<uint64_t>(value));

  if (len <= 0) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  MooreString result = allocateString(len);
  std::memcpy(result.data, buffer, len);
  return result;
}

extern "C" int64_t __moore_string_to_int(MooreString *str) {
  if (!str || !str->data || str->len <= 0)
    return 0;

  // Create a null-terminated copy for parsing
  char *buffer = static_cast<char *>(std::malloc(str->len + 1));
  std::memcpy(buffer, str->data, str->len);
  buffer[str->len] = '\0';

  char *endptr = nullptr;
  int64_t result = std::strtoll(buffer, &endptr, 10);

  std::free(buffer);
  return result;
}

//===----------------------------------------------------------------------===//
// Memory Management
//===----------------------------------------------------------------------===//

extern "C" void __moore_free(void *ptr) {
  std::free(ptr);
}
