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
#include <random>

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

extern "C" void __moore_queue_clear(MooreQueue *queue) {
  // Clear all elements from the queue.
  if (queue->data) {
    std::free(queue->data);
    queue->data = nullptr;
  }
  queue->len = 0;
}

extern "C" void __moore_queue_delete_index(MooreQueue *queue, int32_t index) {
  // Delete element at specified index.
  // TODO: This requires knowing the element size to properly shift elements.
  // For now, this is a placeholder that just marks the operation as done.
  (void)queue;
  (void)index;
}

extern "C" void __moore_queue_push_back(MooreQueue *queue, void *element,
                                        int64_t element_size) {
  if (!queue || !element || element_size <= 0)
    return;

  // Allocate new storage with space for one more element
  int64_t newLen = queue->len + 1;
  void *newData = std::malloc(newLen * element_size);
  if (!newData)
    return;

  // Copy existing elements
  if (queue->data && queue->len > 0) {
    std::memcpy(newData, queue->data, queue->len * element_size);
  }

  // Copy new element to the end
  std::memcpy(static_cast<char *>(newData) + queue->len * element_size,
              element, element_size);

  // Free old data and update queue
  if (queue->data)
    std::free(queue->data);
  queue->data = newData;
  queue->len = newLen;
}

extern "C" void __moore_queue_push_front(MooreQueue *queue, void *element,
                                         int64_t element_size) {
  if (!queue || !element || element_size <= 0)
    return;

  // Allocate new storage with space for one more element
  int64_t newLen = queue->len + 1;
  void *newData = std::malloc(newLen * element_size);
  if (!newData)
    return;

  // Copy new element to the front
  std::memcpy(newData, element, element_size);

  // Copy existing elements after the new one
  if (queue->data && queue->len > 0) {
    std::memcpy(static_cast<char *>(newData) + element_size,
                queue->data, queue->len * element_size);
  }

  // Free old data and update queue
  if (queue->data)
    std::free(queue->data);
  queue->data = newData;
  queue->len = newLen;
}

extern "C" int64_t __moore_queue_pop_back(MooreQueue *queue,
                                          int64_t element_size) {
  if (!queue || !queue->data || queue->len <= 0 || element_size <= 0)
    return 0;

  // Read the last element
  int64_t result = 0;
  void *lastElem = static_cast<char *>(queue->data) +
                   (queue->len - 1) * element_size;
  // Copy up to 8 bytes (size of int64_t)
  std::memcpy(&result, lastElem,
              element_size < 8 ? element_size : 8);

  // Reduce the queue size
  queue->len--;

  // If queue is now empty, free the data
  if (queue->len == 0) {
    std::free(queue->data);
    queue->data = nullptr;
  }
  // Otherwise, we could reallocate to save memory, but for now we keep
  // the existing allocation for simplicity

  return result;
}

extern "C" int64_t __moore_queue_pop_front(MooreQueue *queue,
                                           int64_t element_size) {
  if (!queue || !queue->data || queue->len <= 0 || element_size <= 0)
    return 0;

  // Read the first element
  int64_t result = 0;
  // Copy up to 8 bytes (size of int64_t)
  std::memcpy(&result, queue->data,
              element_size < 8 ? element_size : 8);

  // Reduce the queue size and shift elements
  queue->len--;

  if (queue->len == 0) {
    // Queue is now empty
    std::free(queue->data);
    queue->data = nullptr;
  } else {
    // Shift remaining elements to the front
    std::memmove(queue->data,
                 static_cast<char *>(queue->data) + element_size,
                 queue->len * element_size);
  }

  return result;
}

extern "C" void *__moore_queue_sort(void *queue, int64_t elem_size,
                                    int (*compare)(const void *, const void *)) {
  auto *q = static_cast<MooreQueue *>(queue);

  // Allocate result queue
  auto *result = static_cast<MooreQueue *>(std::malloc(sizeof(MooreQueue)));
  if (!result)
    return nullptr;

  // Handle empty or invalid queue
  if (!q || !q->data || q->len <= 0 || elem_size <= 0) {
    result->data = nullptr;
    result->len = 0;
    return result;
  }

  // Allocate and copy elements to new queue
  int64_t totalSize = q->len * elem_size;
  result->data = std::malloc(totalSize);
  if (!result->data) {
    result->len = 0;
    return result;
  }

  std::memcpy(result->data, q->data, totalSize);
  result->len = q->len;

  // Sort the copied elements using qsort
  std::qsort(result->data, result->len, elem_size, compare);

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
// Streaming Concatenation Operations
//===----------------------------------------------------------------------===//

extern "C" MooreString __moore_stream_concat_strings(MooreQueue *queue,
                                                      bool isRightToLeft) {
  if (!queue || !queue->data || queue->len <= 0) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  // Queue contains MooreString elements
  auto *strings = static_cast<MooreString *>(queue->data);
  int64_t numStrings = queue->len;

  // Calculate total length
  int64_t totalLen = 0;
  for (int64_t i = 0; i < numStrings; ++i) {
    if (strings[i].data && strings[i].len > 0)
      totalLen += strings[i].len;
  }

  if (totalLen == 0) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  MooreString result = allocateString(totalLen);
  char *dst = result.data;

  if (isRightToLeft) {
    // Right-to-left: iterate in reverse order
    for (int64_t i = numStrings - 1; i >= 0; --i) {
      if (strings[i].data && strings[i].len > 0) {
        std::memcpy(dst, strings[i].data, strings[i].len);
        dst += strings[i].len;
      }
    }
  } else {
    // Left-to-right: iterate in normal order
    for (int64_t i = 0; i < numStrings; ++i) {
      if (strings[i].data && strings[i].len > 0) {
        std::memcpy(dst, strings[i].data, strings[i].len);
        dst += strings[i].len;
      }
    }
  }

  return result;
}

extern "C" int64_t __moore_stream_concat_bits(MooreQueue *queue,
                                               int32_t elementBitWidth,
                                               bool isRightToLeft) {
  if (!queue || !queue->data || queue->len <= 0 || elementBitWidth <= 0)
    return 0;

  int64_t numElements = queue->len;
  int64_t result = 0;

  // Calculate bytes per element (round up to whole bytes)
  int32_t bytesPerElement = (elementBitWidth + 7) / 8;

  auto *data = static_cast<uint8_t *>(queue->data);

  if (isRightToLeft) {
    // Right-to-left streaming: pack elements from last to first
    // Each element occupies elementBitWidth bits in the result
    int bitPos = 0;
    for (int64_t i = numElements - 1; i >= 0 && bitPos < 64; --i) {
      // Read the element value (little-endian)
      int64_t elemVal = 0;
      for (int32_t b = 0; b < bytesPerElement && b < 8; ++b) {
        elemVal |= static_cast<int64_t>(data[i * bytesPerElement + b]) << (b * 8);
      }
      // Mask to the actual bit width
      if (elementBitWidth < 64)
        elemVal &= (1LL << elementBitWidth) - 1;

      // Insert into result at current position
      if (bitPos + elementBitWidth <= 64) {
        result |= elemVal << bitPos;
      } else {
        // Partial fit - truncate what doesn't fit
        result |= elemVal << bitPos;
      }
      bitPos += elementBitWidth;
    }
  } else {
    // Left-to-right streaming: pack elements from first to last
    // Elements are placed from MSB to LSB in the result
    int bitPos = 0;
    for (int64_t i = 0; i < numElements && bitPos < 64; ++i) {
      // Read the element value (little-endian)
      int64_t elemVal = 0;
      for (int32_t b = 0; b < bytesPerElement && b < 8; ++b) {
        elemVal |= static_cast<int64_t>(data[i * bytesPerElement + b]) << (b * 8);
      }
      // Mask to the actual bit width
      if (elementBitWidth < 64)
        elemVal &= (1LL << elementBitWidth) - 1;

      // Insert into result at current position
      if (bitPos + elementBitWidth <= 64) {
        result |= elemVal << bitPos;
      } else {
        // Partial fit
        result |= elemVal << bitPos;
      }
      bitPos += elementBitWidth;
    }
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Event Operations
//===----------------------------------------------------------------------===//

extern "C" bool __moore_event_triggered(bool *event) {
  // Check if the event was triggered in the current time slot.
  // Events are represented as boolean flags. The triggered property returns
  // true if the event has been triggered in the current time slot.
  // For now, we simply return the value of the event flag, which represents
  // whether the event was triggered.
  if (!event)
    return false;
  return *event;
}

//===----------------------------------------------------------------------===//
// Simulation Control Operations
//===----------------------------------------------------------------------===//

extern "C" void __moore_wait_condition(int32_t condition) {
  // In a real simulation environment, this would suspend the current process
  // until the condition becomes true. Since simulation timing is not directly
  // supported in the compiled output, this function serves as a placeholder
  // that can be implemented by the simulation runtime.
  //
  // For now, we simply check the condition - if it's already true, we return
  // immediately. In a full simulation environment, if the condition is false,
  // the runtime would need to suspend and re-evaluate when signals change.
  (void)condition;
  // TODO: Implement proper simulation-aware waiting when a simulation
  // scheduler is available.
}

//===----------------------------------------------------------------------===//
// Random Number Generation
//===----------------------------------------------------------------------===//

namespace {
// Thread-local random number generator for reproducible simulation.
// Uses a Mersenne Twister engine which provides good statistical properties.
thread_local std::mt19937 urandomGenerator(std::random_device{}());
thread_local std::mt19937 randomGenerator(std::random_device{}());
} // anonymous namespace

extern "C" uint32_t __moore_urandom(void) {
  // Generate a 32-bit unsigned pseudo-random number.
  return urandomGenerator();
}

extern "C" uint32_t __moore_urandom_seeded(int32_t seed) {
  // Seed the generator and return a random number.
  urandomGenerator.seed(static_cast<uint32_t>(seed));
  return urandomGenerator();
}

extern "C" uint32_t __moore_urandom_range(uint32_t maxval, uint32_t minval) {
  // IEEE 1800-2017 Section 18.13.3: If min > max, swap them.
  if (minval > maxval) {
    uint32_t tmp = minval;
    minval = maxval;
    maxval = tmp;
  }

  // Handle edge case where range is 0
  if (minval == maxval) {
    return minval;
  }

  // Generate a random number in the range [minval, maxval]
  std::uniform_int_distribution<uint32_t> dist(minval, maxval);
  return dist(urandomGenerator);
}

extern "C" int32_t __moore_random(void) {
  // Generate a 32-bit signed random number.
  // $random is supposed to be "truly random" but in practice is implemented
  // as a pseudo-random generator.
  return static_cast<int32_t>(randomGenerator());
}

extern "C" int32_t __moore_random_seeded(int32_t seed) {
  // Seed the generator and return a random number.
  randomGenerator.seed(static_cast<uint32_t>(seed));
  return static_cast<int32_t>(randomGenerator());
}

//===----------------------------------------------------------------------===//
// Dynamic Cast / RTTI Operations
//===----------------------------------------------------------------------===//

extern "C" bool __moore_dyn_cast_check(int32_t srcTypeId, int32_t targetTypeId,
                                       int32_t inheritanceDepth) {
  // Dynamic cast check: verifies if a source object can be cast to a target type.
  //
  // Type IDs are assigned such that:
  // - Each class gets a unique type ID
  // - A derived class's type ID is related to its base class's type ID
  //
  // For a simple implementation, we use the following scheme:
  // - Type IDs are assigned sequentially per class
  // - The cast succeeds if the source type ID matches the target type ID exactly,
  //   OR if the source is a derived type that includes the target in its hierarchy.
  //
  // The inheritanceDepth parameter helps determine if srcTypeId could be a
  // valid derived type of targetTypeId. In a simple scheme where type IDs
  // are assigned sequentially with derived classes getting higher IDs than
  // their bases, a valid downcast requires:
  //   srcTypeId >= targetTypeId (derived classes have higher or equal IDs)
  //
  // For now, we implement a simple check:
  // - Exact match: srcTypeId == targetTypeId (same type)
  // - Derived check: srcTypeId > targetTypeId (src could be derived from target)
  //
  // Note: A full implementation would use a class hierarchy table to verify
  // the exact inheritance relationship. This simplified version assumes
  // the compiler assigns type IDs such that derived classes always have
  // higher IDs than their base classes (topological ordering).

  // Null type IDs (0) indicate uninitialized or invalid - always fail
  if (srcTypeId == 0 || targetTypeId == 0)
    return false;

  // Same type - always succeeds
  if (srcTypeId == targetTypeId)
    return true;

  // For downcasting: source must be a derived type (higher or equal type ID)
  // This works because type IDs are assigned in topological order (base first)
  // A derived class always has a type ID >= its base class
  //
  // However, this simple check can have false positives (sibling classes).
  // For production use, a proper class hierarchy lookup table should be used.
  // For now, we accept that srcTypeId >= targetTypeId indicates a potential
  // derived relationship.
  (void)inheritanceDepth; // Reserved for future hierarchy depth checking

  return srcTypeId >= targetTypeId;
}

//===----------------------------------------------------------------------===//
// Memory Management
//===----------------------------------------------------------------------===//

extern "C" void __moore_free(void *ptr) {
  std::free(ptr);
}
