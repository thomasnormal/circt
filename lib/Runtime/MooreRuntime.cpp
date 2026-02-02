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
#include <atomic>
#include <cctype>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <set>
#include <thread>
#include <unordered_map>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <unistd.h>
#endif

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

std::string convertGlobToRegex(const std::string &pattern, bool withBrackets) {
  std::string regex;
  for (size_t i = 0; i < pattern.size(); ++i) {
    char c = pattern[i];
    if (withBrackets && c == '[') {
      regex.push_back('[');
      for (++i; i < pattern.size(); ++i) {
        char b = pattern[i];
        regex.push_back(b);
        if (b == ']')
          break;
      }
      continue;
    }
    switch (c) {
    case '*':
      regex += ".*";
      break;
    case '?':
      regex += ".";
      break;
    case '.':
    case '^':
    case '$':
    case '+':
    case '(':
    case ')':
    case '[':
    case ']':
    case '{':
    case '}':
    case '|':
    case '\\':
      regex += '\\';
      regex += c;
      break;
    default:
      regex += c;
      break;
    }
  }
  return regex;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Queue Operations
//===----------------------------------------------------------------------===//

// NOTE: __moore_queue_max and __moore_queue_min are deprecated.
// Use __moore_array_max and __moore_array_min instead, which take
// element size and signedness parameters for proper comparison.
// These are kept for backward compatibility but return empty queues.

extern "C" MooreQueue __moore_queue_max(MooreQueue *queue) {
  // DEPRECATED: Use __moore_array_max instead.
  MooreQueue result = {nullptr, 0};
  (void)queue;
  return result;
}

extern "C" MooreQueue __moore_queue_min(MooreQueue *queue) {
  // DEPRECATED: Use __moore_array_min instead.
  MooreQueue result = {nullptr, 0};
  (void)queue;
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

extern "C" void __moore_queue_delete_index(MooreQueue *queue, int32_t index,
                                           int64_t element_size) {
  // Delete element at specified index.
  // SystemVerilog semantics: delete(index) removes the element at that index
  // and shifts all subsequent elements down by one position.
  if (!queue || !queue->data || element_size <= 0)
    return;

  // Bounds check: index must be valid
  if (index < 0 || index >= queue->len)
    return;

  // If this is the last element, just shrink
  if (queue->len == 1) {
    std::free(queue->data);
    queue->data = nullptr;
    queue->len = 0;
    return;
  }

  // Allocate new storage with one fewer element
  int64_t newLen = queue->len - 1;
  void *newData = std::malloc(newLen * element_size);
  if (!newData)
    return;

  char *src = static_cast<char *>(queue->data);
  char *dst = static_cast<char *>(newData);

  // Copy elements before the deleted index
  if (index > 0) {
    std::memcpy(dst, src, index * element_size);
  }

  // Copy elements after the deleted index
  if (index < queue->len - 1) {
    std::memcpy(dst + index * element_size,
                src + (index + 1) * element_size,
                (queue->len - index - 1) * element_size);
  }

  // Free old data and update queue
  std::free(queue->data);
  queue->data = newData;
  queue->len = newLen;
}

extern "C" void __moore_queue_insert(MooreQueue *queue, int32_t index,
                                     void *element, int64_t element_size) {
  // Insert element at specified index.
  // SystemVerilog semantics: insert(index, item) inserts the item at the
  // specified index, shifting all subsequent elements up by one position.
  // If index < 0, it's treated as 0. If index >= size, the item is appended.
  if (!queue || !element || element_size <= 0)
    return;

  // Clamp index to valid range
  if (index < 0)
    index = 0;
  if (index > queue->len)
    index = static_cast<int32_t>(queue->len);

  // Allocate new storage with space for one more element
  int64_t newLen = queue->len + 1;
  void *newData = std::malloc(newLen * element_size);
  if (!newData)
    return;

  char *src = static_cast<char *>(queue->data);
  char *dst = static_cast<char *>(newData);

  // Copy elements before the insertion index
  if (index > 0 && queue->data) {
    std::memcpy(dst, src, index * element_size);
  }

  // Copy the new element at the insertion index
  std::memcpy(dst + index * element_size, element, element_size);

  // Copy elements after the insertion index
  if (queue->data && index < queue->len) {
    std::memcpy(dst + (index + 1) * element_size,
                src + index * element_size,
                (queue->len - index) * element_size);
  }

  // Free old data and update queue
  if (queue->data)
    std::free(queue->data);
  queue->data = newData;
  queue->len = newLen;
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

extern "C" MooreQueue __moore_queue_slice(MooreQueue *queue, int64_t start,
                                          int64_t end, int64_t element_size) {
  MooreQueue result = {nullptr, 0};
  if (!queue || !queue->data || queue->len <= 0 || element_size <= 0)
    return result;

  int64_t len = queue->len;
  if (start < 0)
    start = 0;
  if (end < 0)
    return result;
  if (start >= len)
    return result;
  if (end >= len)
    end = len - 1;
  if (end < start)
    return result;

  int64_t sliceLen = end - start + 1;
  if (sliceLen <= 0)
    return result;

  void *data = std::malloc(sliceLen * element_size);
  if (!data)
    return result;

  std::memcpy(data,
              static_cast<char *>(queue->data) + start * element_size,
              sliceLen * element_size);
  result.data = data;
  result.len = sliceLen;
  return result;
}

extern "C" MooreQueue __moore_queue_concat(MooreQueue *queues, int64_t count,
                                           int64_t element_size) {
  MooreQueue result = {nullptr, 0};
  if (!queues || count <= 0 || element_size <= 0)
    return result;

  int64_t totalLen = 0;
  for (int64_t i = 0; i < count; ++i) {
    if (queues[i].len > 0)
      totalLen += queues[i].len;
  }
  if (totalLen <= 0)
    return result;

  void *data = std::malloc(totalLen * element_size);
  if (!data)
    return result;

  int64_t offset = 0;
  for (int64_t i = 0; i < count; ++i) {
    const auto &q = queues[i];
    if (!q.data || q.len <= 0)
      continue;
    std::memcpy(static_cast<char *>(data) + offset * element_size, q.data,
                q.len * element_size);
    offset += q.len;
  }

  result.data = data;
  result.len = totalLen;
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

namespace {
// Helper function forward declaration
uint64_t readElementValueUnsigned(void *element, int64_t elementSize);

thread_local int64_t queueSortElemSize = 0;

int compareQueueElemDesc(const void *a, const void *b) {
  uint64_t va = readElementValueUnsigned(const_cast<void*>(a), queueSortElemSize);
  uint64_t vb = readElementValueUnsigned(const_cast<void*>(b), queueSortElemSize);
  if (va < vb)
    return 1;
  if (va > vb)
    return -1;
  return 0;
}
} // namespace

extern "C" void __moore_queue_rsort(MooreQueue *queue, int64_t elem_size) {
  if (!queue || !queue->data || queue->len <= 1 || elem_size <= 0 ||
      elem_size > 8)
    return;

  queueSortElemSize = elem_size;
  std::qsort(queue->data, queue->len, elem_size, compareQueueElemDesc);
}

extern "C" void __moore_queue_shuffle(MooreQueue *queue, int64_t elem_size) {
  if (!queue || !queue->data || queue->len <= 1 || elem_size <= 0)
    return;

  auto *data = static_cast<char *>(queue->data);
  std::vector<char> temp(static_cast<size_t>(elem_size));
  for (int64_t i = queue->len - 1; i > 0; --i) {
    int64_t j = std::rand() % (i + 1);
    if (i == j)
      continue;

    std::memcpy(temp.data(), data + i * elem_size,
                static_cast<size_t>(elem_size));
    std::memcpy(data + i * elem_size, data + j * elem_size,
                static_cast<size_t>(elem_size));
    std::memcpy(data + j * elem_size, temp.data(),
                static_cast<size_t>(elem_size));
  }
}

extern "C" void __moore_queue_reverse(MooreQueue *queue, int64_t elem_size) {
  if (!queue || !queue->data || queue->len <= 1 || elem_size <= 0)
    return;

  auto *data = static_cast<char *>(queue->data);
  std::vector<char> temp(static_cast<size_t>(elem_size));
  int64_t left = 0;
  int64_t right = queue->len - 1;
  while (left < right) {
    // Swap elements at left and right
    std::memcpy(temp.data(), data + left * elem_size,
                static_cast<size_t>(elem_size));
    std::memcpy(data + left * elem_size, data + right * elem_size,
                static_cast<size_t>(elem_size));
    std::memcpy(data + right * elem_size, temp.data(),
                static_cast<size_t>(elem_size));
    ++left;
    --right;
  }
}

extern "C" void __moore_queue_pop_back_ptr(MooreQueue *queue, void *result_ptr,
                                            int64_t element_size) {
  if (!queue || !queue->data || queue->len <= 0 || element_size <= 0 ||
      !result_ptr)
    return;

  // Copy the last element to the result buffer
  void *lastElem = static_cast<char *>(queue->data) +
                   (queue->len - 1) * element_size;
  std::memcpy(result_ptr, lastElem, element_size);

  // Reduce the queue size
  queue->len--;

  // If queue is now empty, free the data
  if (queue->len == 0) {
    std::free(queue->data);
    queue->data = nullptr;
  }
}

extern "C" void __moore_queue_pop_front_ptr(MooreQueue *queue, void *result_ptr,
                                             int64_t element_size) {
  if (!queue || !queue->data || queue->len <= 0 || element_size <= 0 ||
      !result_ptr)
    return;

  // Copy the first element to the result buffer
  std::memcpy(result_ptr, queue->data, element_size);

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
}

extern "C" int64_t __moore_queue_size(MooreQueue *queue) {
  if (!queue)
    return 0;
  return queue->len;
}

extern "C" MooreQueue __moore_queue_unique(MooreQueue *queue) {
  MooreQueue result = {nullptr, 0};

  // This simplified version assumes 8-byte elements (pointers/int64).
  // For full support, element size should be passed as a parameter.
  // The MooreToCore should be updated to use __moore_array_unique instead.
  if (!queue || !queue->data || queue->len <= 0)
    return result;

  // Default to 8-byte element size (common for pointers and int64)
  int64_t elementSize = 8;

  auto *data = static_cast<char *>(queue->data);
  int64_t numElements = queue->len;

  // For each element, check if it's already in the result
  for (int64_t i = 0; i < numElements; ++i) {
    void *element = data + i * elementSize;
    bool found = false;

    // Check if this element is already in the result
    auto *resultData = static_cast<char *>(result.data);
    for (int64_t j = 0; j < result.len; ++j) {
      if (std::memcmp(resultData + j * elementSize, element, elementSize) ==
          0) {
        found = true;
        break;
      }
    }

    if (!found) {
      // Append element to result
      int64_t newLen = result.len + 1;
      void *newData = std::realloc(result.data, newLen * elementSize);
      if (!newData)
        return result;
      std::memcpy(static_cast<char *>(newData) + result.len * elementSize,
                  element, elementSize);
      result.data = newData;
      result.len = newLen;
    }
  }

  return result;
}

namespace {
thread_local int64_t queueSortInplaceElemSize = 0;

int compareQueueElemAsc(const void *a, const void *b) {
  uint64_t va = 0, vb = 0;
  if (queueSortInplaceElemSize > 0 && queueSortInplaceElemSize <= 8) {
    std::memcpy(&va, a, queueSortInplaceElemSize);
    std::memcpy(&vb, b, queueSortInplaceElemSize);
  }
  if (va < vb)
    return -1;
  if (va > vb)
    return 1;
  return 0;
}
} // namespace

extern "C" void __moore_queue_sort_inplace(MooreQueue *queue, int64_t elem_size) {
  if (!queue || !queue->data || queue->len <= 1 || elem_size <= 0 ||
      elem_size > 8)
    return;

  queueSortInplaceElemSize = elem_size;
  std::qsort(queue->data, queue->len, elem_size, compareQueueElemAsc);
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

namespace {

/// Internal structure for associative arrays with string keys.
/// Uses std::map<std::string, std::vector<uint8_t>> to store the data.
struct StringKeyAssocArray {
  std::map<std::string, std::vector<uint8_t>> data;
  int32_t valueSize;
  // Iterator state for first/next/last/prev
  std::map<std::string, std::vector<uint8_t>>::iterator currentIter;
  bool iterValid = false;
};

/// Internal structure for associative arrays with integer keys.
/// Uses std::map<int64_t, std::vector<uint8_t>> to store the data.
struct IntKeyAssocArray {
  std::map<int64_t, std::vector<uint8_t>> data;
  int32_t keySize;
  int32_t valueSize;
  // Iterator state for first/next/last/prev
  std::map<int64_t, std::vector<uint8_t>>::iterator currentIter;
  bool iterValid = false;
};

/// Helper to read an integer key from memory based on key size.
int64_t readIntKey(void *key, int32_t keySize) {
  switch (keySize) {
  case 1:
    return *static_cast<int8_t *>(key);
  case 2:
    return *static_cast<int16_t *>(key);
  case 4:
    return *static_cast<int32_t *>(key);
  case 8:
    return *static_cast<int64_t *>(key);
  default:
    return 0;
  }
}

/// Helper to write an integer key to memory based on key size.
void writeIntKey(void *key, int64_t value, int32_t keySize) {
  switch (keySize) {
  case 1:
    *static_cast<int8_t *>(key) = static_cast<int8_t>(value);
    break;
  case 2:
    *static_cast<int16_t *>(key) = static_cast<int16_t>(value);
    break;
  case 4:
    *static_cast<int32_t *>(key) = static_cast<int32_t>(value);
    break;
  case 8:
    *static_cast<int64_t *>(key) = value;
    break;
  }
}

// Use the AssocArrayType and AssocArrayHeader from the header file.
// These are C-style types for ABI compatibility.

} // anonymous namespace

extern "C" void *__moore_assoc_create(int32_t key_size, int32_t value_size) {
  auto *header = new AssocArrayHeader;
  if (key_size == 0) {
    // String-keyed associative array
    auto *arr = new StringKeyAssocArray;
    arr->valueSize = value_size;
    header->type = AssocArrayType_StringKey;
    header->array = arr;
  } else {
    // Integer-keyed associative array
    auto *arr = new IntKeyAssocArray;
    arr->keySize = key_size;
    arr->valueSize = value_size;
    header->type = AssocArrayType_IntKey;
    header->array = arr;
  }
  return header;
}

extern "C" int64_t __moore_assoc_size(void *array) {
  if (!array)
    return 0;
  auto *header = static_cast<AssocArrayHeader *>(array);
  if (header->type == AssocArrayType_StringKey) {
    auto *arr = static_cast<StringKeyAssocArray *>(header->array);
    return static_cast<int64_t>(arr->data.size());
  } else {
    auto *arr = static_cast<IntKeyAssocArray *>(header->array);
    return static_cast<int64_t>(arr->data.size());
  }
}

/// Check if an unpacked array contains a value.
/// This supports static unpacked arrays, dynamic arrays, and queues.
/// Uses byte-wise comparison for the element values.
/// @param arr Pointer to the array data (contiguous elements)
/// @param numElems Number of elements in the array
/// @param value Pointer to the value to search for
/// @param elemSize Size of each element in bytes
/// @return true if the value is found in the array, false otherwise
extern "C" bool __moore_array_contains(void *arr, int64_t numElems,
                                       void *value, int64_t elemSize) {
  if (!arr || !value || numElems <= 0 || elemSize <= 0)
    return false;

  auto *arrData = static_cast<const uint8_t *>(arr);
  auto *valueData = static_cast<const uint8_t *>(value);

  for (int64_t i = 0; i < numElems; ++i) {
    const uint8_t *elemPtr = arrData + (i * elemSize);
    if (std::memcmp(elemPtr, valueData, static_cast<size_t>(elemSize)) == 0)
      return true;
  }
  return false;
}

extern "C" void __moore_assoc_delete(void *array) {
  if (!array)
    return;
  auto *header = static_cast<AssocArrayHeader *>(array);
  if (header->type == AssocArrayType_StringKey) {
    auto *arr = static_cast<StringKeyAssocArray *>(header->array);
    arr->data.clear();
  } else {
    auto *arr = static_cast<IntKeyAssocArray *>(header->array);
    arr->data.clear();
  }
}

extern "C" void __moore_assoc_delete_key(void *array, void *key) {
  if (!array || !key)
    return;
  auto *header = static_cast<AssocArrayHeader *>(array);
  if (header->type == AssocArrayType_StringKey) {
    auto *arr = static_cast<StringKeyAssocArray *>(header->array);
    auto *strKey = static_cast<MooreString *>(key);
    if (strKey->data) {
      std::string keyStr(strKey->data, strKey->len);
      arr->data.erase(keyStr);
    }
  } else {
    auto *arr = static_cast<IntKeyAssocArray *>(header->array);
    int64_t intKey = readIntKey(key, arr->keySize);
    arr->data.erase(intKey);
  }
}

extern "C" bool __moore_assoc_first(void *array, void *key_out) {
  if (!array || !key_out)
    return false;
  auto *header = static_cast<AssocArrayHeader *>(array);
  if (header->type == AssocArrayType_StringKey) {
    auto *arr = static_cast<StringKeyAssocArray *>(header->array);
    if (arr->data.empty())
      return false;
    arr->currentIter = arr->data.begin();
    arr->iterValid = true;
    // Copy the key string to the output
    auto *strOut = static_cast<MooreString *>(key_out);
    const std::string &keyStr = arr->currentIter->first;
    // Free existing string data if any
    if (strOut->data)
      std::free(strOut->data);
    strOut->data = static_cast<char *>(std::malloc(keyStr.size()));
    strOut->len = keyStr.size();
    std::memcpy(strOut->data, keyStr.data(), keyStr.size());
    return true;
  } else {
    auto *arr = static_cast<IntKeyAssocArray *>(header->array);
    if (arr->data.empty())
      return false;
    arr->currentIter = arr->data.begin();
    arr->iterValid = true;
    writeIntKey(key_out, arr->currentIter->first, arr->keySize);
    return true;
  }
}

extern "C" bool __moore_assoc_next(void *array, void *key_ref) {
  if (!array || !key_ref)
    return false;
  auto *header = static_cast<AssocArrayHeader *>(array);
  if (header->type == AssocArrayType_StringKey) {
    auto *arr = static_cast<StringKeyAssocArray *>(header->array);
    if (arr->data.empty())
      return false;
    // Get current key from key_ref
    auto *strRef = static_cast<MooreString *>(key_ref);
    if (!strRef->data)
      return false;
    std::string currentKey(strRef->data, strRef->len);
    // Find the next key after the current one
    auto it = arr->data.find(currentKey);
    if (it == arr->data.end())
      return false;
    ++it;
    if (it == arr->data.end())
      return false;
    // Update the key_ref with the next key
    const std::string &nextKey = it->first;
    if (strRef->data)
      std::free(strRef->data);
    strRef->data = static_cast<char *>(std::malloc(nextKey.size()));
    strRef->len = nextKey.size();
    std::memcpy(strRef->data, nextKey.data(), nextKey.size());
    return true;
  } else {
    auto *arr = static_cast<IntKeyAssocArray *>(header->array);
    if (arr->data.empty())
      return false;
    int64_t currentKey = readIntKey(key_ref, arr->keySize);
    auto it = arr->data.find(currentKey);
    if (it == arr->data.end())
      return false;
    ++it;
    if (it == arr->data.end())
      return false;
    writeIntKey(key_ref, it->first, arr->keySize);
    return true;
  }
}

extern "C" bool __moore_assoc_last(void *array, void *key_out) {
  if (!array || !key_out)
    return false;
  auto *header = static_cast<AssocArrayHeader *>(array);
  if (header->type == AssocArrayType_StringKey) {
    auto *arr = static_cast<StringKeyAssocArray *>(header->array);
    if (arr->data.empty())
      return false;
    auto it = arr->data.end();
    --it;
    // Copy the key string to the output
    auto *strOut = static_cast<MooreString *>(key_out);
    const std::string &keyStr = it->first;
    if (strOut->data)
      std::free(strOut->data);
    strOut->data = static_cast<char *>(std::malloc(keyStr.size()));
    strOut->len = keyStr.size();
    std::memcpy(strOut->data, keyStr.data(), keyStr.size());
    return true;
  } else {
    auto *arr = static_cast<IntKeyAssocArray *>(header->array);
    if (arr->data.empty())
      return false;
    auto it = arr->data.end();
    --it;
    writeIntKey(key_out, it->first, arr->keySize);
    return true;
  }
}

extern "C" bool __moore_assoc_prev(void *array, void *key_ref) {
  if (!array || !key_ref)
    return false;
  auto *header = static_cast<AssocArrayHeader *>(array);
  if (header->type == AssocArrayType_StringKey) {
    auto *arr = static_cast<StringKeyAssocArray *>(header->array);
    if (arr->data.empty())
      return false;
    auto *strRef = static_cast<MooreString *>(key_ref);
    if (!strRef->data)
      return false;
    std::string currentKey(strRef->data, strRef->len);
    auto it = arr->data.find(currentKey);
    if (it == arr->data.end() || it == arr->data.begin())
      return false;
    --it;
    const std::string &prevKey = it->first;
    if (strRef->data)
      std::free(strRef->data);
    strRef->data = static_cast<char *>(std::malloc(prevKey.size()));
    strRef->len = prevKey.size();
    std::memcpy(strRef->data, prevKey.data(), prevKey.size());
    return true;
  } else {
    auto *arr = static_cast<IntKeyAssocArray *>(header->array);
    if (arr->data.empty())
      return false;
    int64_t currentKey = readIntKey(key_ref, arr->keySize);
    auto it = arr->data.find(currentKey);
    if (it == arr->data.end() || it == arr->data.begin())
      return false;
    --it;
    writeIntKey(key_ref, it->first, arr->keySize);
    return true;
  }
}

extern "C" int32_t __moore_assoc_exists(void *array, void *key) {
  if (!array || !key)
    return 0;
  auto *header = static_cast<AssocArrayHeader *>(array);
  if (header->type == AssocArrayType_StringKey) {
    auto *arr = static_cast<StringKeyAssocArray *>(header->array);
    auto *strKey = static_cast<MooreString *>(key);
    std::string keyStr;
    if (strKey->data && strKey->len > 0)
      keyStr = std::string(strKey->data, strKey->len);
    return arr->data.find(keyStr) != arr->data.end() ? 1 : 0;
  } else {
    auto *arr = static_cast<IntKeyAssocArray *>(header->array);
    int64_t intKey = readIntKey(key, arr->keySize);
    return arr->data.find(intKey) != arr->data.end() ? 1 : 0;
  }
}

extern "C" void *__moore_assoc_get_ref(void *array, void *key,
                                       int32_t value_size) {
  if (!array || !key)
    return nullptr;
  auto *header = static_cast<AssocArrayHeader *>(array);
  if (header->type == AssocArrayType_StringKey) {
    auto *arr = static_cast<StringKeyAssocArray *>(header->array);
    auto *strKey = static_cast<MooreString *>(key);
    std::string keyStr;
    if (strKey->data && strKey->len > 0)
      keyStr = std::string(strKey->data, strKey->len);
    // Insert or find the element
    auto &valueVec = arr->data[keyStr];
    if (valueVec.empty()) {
      // Initialize with zeros
      valueVec.resize(value_size, 0);
    }
    return valueVec.data();
  } else {
    auto *arr = static_cast<IntKeyAssocArray *>(header->array);
    int64_t intKey = readIntKey(key, arr->keySize);
    auto &valueVec = arr->data[intKey];
    if (valueVec.empty()) {
      valueVec.resize(value_size, 0);
    }
    return valueVec.data();
  }
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

// str.compare(s) - lexicographic string comparison (case-sensitive)
// IEEE 1800-2017 Section 6.16.8
extern "C" int32_t __moore_string_compare(MooreString *lhs, MooreString *rhs) {
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

// str.icompare(s) - lexicographic string comparison (case-insensitive)
// IEEE 1800-2017 Section 6.16.8
extern "C" int32_t __moore_string_icompare(MooreString *lhs, MooreString *rhs) {
  // Handle null/empty cases
  bool lhsEmpty = !lhs || !lhs->data || lhs->len <= 0;
  bool rhsEmpty = !rhs || !rhs->data || rhs->len <= 0;

  if (lhsEmpty && rhsEmpty)
    return 0;
  if (lhsEmpty)
    return -1;
  if (rhsEmpty)
    return 1;

  // Compare up to the minimum length (case-insensitive)
  int64_t minLen = std::min(lhs->len, rhs->len);
  for (int64_t i = 0; i < minLen; ++i) {
    int c1 = std::tolower(static_cast<unsigned char>(lhs->data[i]));
    int c2 = std::tolower(static_cast<unsigned char>(rhs->data[i]));
    if (c1 != c2)
      return c1 - c2;
  }

  // If equal up to minLen, the shorter string is "less"
  if (lhs->len < rhs->len)
    return -1;
  if (lhs->len > rhs->len)
    return 1;
  return 0;
}

extern "C" MooreString __moore_string_replicate(MooreString *str, int32_t count) {
  // Handle null/empty string or non-positive count
  if (!str || !str->data || str->len <= 0 || count <= 0) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  int64_t totalLen = str->len * static_cast<int64_t>(count);
  MooreString result = allocateString(totalLen);

  // Copy the string count times
  char *dst = result.data;
  for (int32_t i = 0; i < count; ++i) {
    std::memcpy(dst, str->data, str->len);
    dst += str->len;
  }

  return result;
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

extern "C" MooreString __moore_packed_string_to_string(int64_t value) {
  // SystemVerilog semantics: integers used as strings have their bytes unpacked
  // from most significant to least significant (big-endian byte order).
  // E.g., 0x48444C5F544F50 ("HDL_TOP" packed) -> "HDL_TOP"
  //
  // The value is stored in native endianness but represents a big-endian
  // packed string where the first character is in the MSB position.

  if (value == 0) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  // Extract bytes from the value (big-endian packed string)
  // First, find the actual length by counting non-zero bytes from MSB
  char buffer[8];
  int len = 0;
  uint64_t uval = static_cast<uint64_t>(value);

  // Extract bytes from MSB to LSB
  for (int i = 7; i >= 0; --i) {
    char c = static_cast<char>((uval >> (i * 8)) & 0xFF);
    if (c != 0 || len > 0) {
      buffer[len++] = c;
    }
  }

  if (len == 0) {
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

extern "C" void __moore_stream_unpack_bits(MooreQueue *array, int64_t sourceBits,
                                            int32_t elementBitWidth,
                                            bool isRightToLeft) {
  if (!array || elementBitWidth <= 0)
    return;

  // Calculate how many elements we can extract from sourceBits
  // For a 64-bit source, that's 64 / elementBitWidth elements max
  int64_t numElements = 64 / elementBitWidth;
  if (numElements <= 0)
    numElements = 1;

  // Calculate bytes per element (round up to whole bytes)
  int32_t bytesPerElement = (elementBitWidth + 7) / 8;

  // Resize the array to hold the elements
  int64_t newSize = numElements * bytesPerElement;

  // Reallocate if needed
  if (array->data) {
    free(array->data);
  }
  array->data = malloc(newSize);
  if (!array->data) {
    array->len = 0;
    return;
  }
  array->len = numElements;

  auto *data = static_cast<uint8_t *>(array->data);
  memset(data, 0, newSize);

  // Mask for extracting element bits
  int64_t elementMask = (elementBitWidth < 64)
                            ? ((1LL << elementBitWidth) - 1)
                            : static_cast<int64_t>(-1);

  if (isRightToLeft) {
    // Right-to-left: extract from LSB to MSB, store from last to first
    int bitPos = 0;
    for (int64_t i = numElements - 1; i >= 0 && bitPos < 64; --i) {
      int64_t elemVal = (sourceBits >> bitPos) & elementMask;
      // Store element (little-endian)
      for (int32_t b = 0; b < bytesPerElement && b < 8; ++b) {
        data[i * bytesPerElement + b] =
            static_cast<uint8_t>((elemVal >> (b * 8)) & 0xFF);
      }
      bitPos += elementBitWidth;
    }
  } else {
    // Left-to-right: extract from LSB to MSB, store from first to last
    int bitPos = 0;
    for (int64_t i = 0; i < numElements && bitPos < 64; ++i) {
      int64_t elemVal = (sourceBits >> bitPos) & elementMask;
      // Store element (little-endian)
      for (int32_t b = 0; b < bytesPerElement && b < 8; ++b) {
        data[i * bytesPerElement + b] =
            static_cast<uint8_t>((elemVal >> (b * 8)) & 0xFF);
      }
      bitPos += elementBitWidth;
    }
  }
}

/// Mixed streaming concatenation with arbitrary-width prefix and suffix.
/// This function handles streaming concatenation where static prefix and suffix
/// values are stored in byte arrays (to support widths > 64 bits).
///
/// Parameters:
/// - prefixData: Pointer to byte array containing prefix bits (little-endian)
/// - prefixBits: Number of bits in prefix
/// - arrayPtr: Pointer to the dynamic array/queue
/// - elemWidth: Bit width of each element in the dynamic array
/// - suffixData: Pointer to byte array containing suffix bits (little-endian)
/// - suffixBits: Number of bits in suffix
/// - sliceSize: Streaming slice size (usually 8 for byte streaming)
/// - isRightToLeft: Direction of streaming
///
/// Returns a new queue containing all the concatenated bits.
extern "C" MooreQueue __moore_stream_concat_mixed(uint8_t *prefixData,
                                                   int32_t prefixBits,
                                                   MooreQueue *arrayPtr,
                                                   int32_t elemWidth,
                                                   uint8_t *suffixData,
                                                   int32_t suffixBits,
                                                   int32_t sliceSize,
                                                   bool isRightToLeft) {
  MooreQueue result = {nullptr, 0};

  // Calculate total bits
  int64_t arrayBits = 0;
  if (arrayPtr && arrayPtr->data && arrayPtr->len > 0) {
    arrayBits = arrayPtr->len * elemWidth;
  }
  int64_t totalBits = prefixBits + arrayBits + suffixBits;

  if (totalBits == 0)
    return result;

  // Determine output element size based on slice size (usually byte-sized)
  int32_t outputElemWidth = sliceSize > 0 ? sliceSize : 8;
  int32_t outputElemBytes = (outputElemWidth + 7) / 8;

  // Calculate number of output elements (round up)
  int64_t numOutputElems = (totalBits + outputElemWidth - 1) / outputElemWidth;

  // Allocate output queue
  result.data = malloc(numOutputElems * outputElemBytes);
  if (!result.data)
    return result;
  result.len = numOutputElems;
  memset(result.data, 0, numOutputElems * outputElemBytes);

  auto *outData = static_cast<uint8_t *>(result.data);

  // Helper lambda to copy bits from source to destination
  auto copyBits = [](uint8_t *dst, int64_t dstBitOffset, const uint8_t *src,
                     int64_t srcBitOffset, int64_t numBits) {
    for (int64_t i = 0; i < numBits; ++i) {
      int64_t srcByteIdx = (srcBitOffset + i) / 8;
      int64_t srcBitIdx = (srcBitOffset + i) % 8;
      int64_t dstByteIdx = (dstBitOffset + i) / 8;
      int64_t dstBitIdx = (dstBitOffset + i) % 8;

      uint8_t bit = (src[srcByteIdx] >> srcBitIdx) & 1;
      dst[dstByteIdx] |= (bit << dstBitIdx);
    }
  };

  // Get array data
  auto *arrData = arrayPtr ? static_cast<uint8_t *>(arrayPtr->data) : nullptr;
  int64_t arrLen = arrayPtr ? arrayPtr->len : 0;
  int32_t elemBytes = (elemWidth + 7) / 8;

  if (isRightToLeft) {
    // Right-to-left streaming: process in reverse order
    // Order: suffix first (reversed), then array (reversed), then prefix (reversed)
    int64_t outBitPos = 0;

    // Copy suffix bits (reversed)
    if (suffixBits > 0 && suffixData) {
      copyBits(outData, outBitPos, suffixData, 0, suffixBits);
      outBitPos += suffixBits;
    }

    // Copy array elements (reversed)
    if (arrLen > 0 && arrData) {
      for (int64_t i = arrLen - 1; i >= 0; --i) {
        copyBits(outData, outBitPos, arrData + i * elemBytes, 0, elemWidth);
        outBitPos += elemWidth;
      }
    }

    // Copy prefix bits (reversed)
    if (prefixBits > 0 && prefixData) {
      copyBits(outData, outBitPos, prefixData, 0, prefixBits);
      outBitPos += prefixBits;
    }
  } else {
    // Left-to-right streaming: process in normal order
    // Order: prefix first, then array, then suffix
    int64_t outBitPos = 0;

    // Copy prefix bits
    if (prefixBits > 0 && prefixData) {
      copyBits(outData, outBitPos, prefixData, 0, prefixBits);
      outBitPos += prefixBits;
    }

    // Copy array elements
    if (arrLen > 0 && arrData) {
      for (int64_t i = 0; i < arrLen; ++i) {
        copyBits(outData, outBitPos, arrData + i * elemBytes, 0, elemWidth);
        outBitPos += elemWidth;
      }
    }

    // Copy suffix bits
    if (suffixBits > 0 && suffixData) {
      copyBits(outData, outBitPos, suffixData, 0, suffixBits);
      outBitPos += suffixBits;
    }
  }

  return result;
}

/// Result structure for mixed streaming unpack extraction.
/// Contains the extracted prefix, middle (dynamic array), and suffix.
struct StreamUnpackMixedResult {
  uint8_t *prefixData;  // Extracted prefix bits (caller must free)
  int32_t prefixBytes;  // Number of bytes in prefix
  MooreQueue middle;    // Extracted middle queue
  uint8_t *suffixData;  // Extracted suffix bits (caller must free)
  int32_t suffixBytes;  // Number of bytes in suffix
};

/// Mixed streaming unpack extraction with arbitrary-width prefix and suffix.
/// This function extracts bits from a source queue into prefix, middle, and
/// suffix portions for streaming unpack operations.
///
/// Parameters:
/// - srcPtr: Pointer to the source queue
/// - prefixBits: Number of bits to extract for prefix
/// - elemWidth: Bit width of each element in the dynamic array
/// - suffixBits: Number of bits to extract for suffix
/// - sliceSize: Streaming slice size
/// - isRightToLeft: Direction of streaming
///
/// Returns a result structure with extracted prefix, middle queue, and suffix.
extern "C" StreamUnpackMixedResult
__moore_stream_unpack_mixed_extract(MooreQueue *srcPtr, int32_t prefixBits,
                                     int32_t elemWidth, int32_t suffixBits,
                                     int32_t sliceSize, bool isRightToLeft) {
  StreamUnpackMixedResult result = {nullptr, 0, {nullptr, 0}, nullptr, 0};

  if (!srcPtr || !srcPtr->data || srcPtr->len <= 0)
    return result;

  auto *srcData = static_cast<uint8_t *>(srcPtr->data);
  int64_t srcLen = srcPtr->len;

  // Calculate total source bits
  int64_t totalSrcBits = srcLen * sliceSize;

  // Calculate middle bits (what's left after prefix and suffix)
  int64_t middleBits = totalSrcBits - prefixBits - suffixBits;
  if (middleBits < 0)
    middleBits = 0;

  // Allocate prefix buffer
  int32_t prefixBytes = (prefixBits + 7) / 8;
  if (prefixBytes > 0) {
    result.prefixData = static_cast<uint8_t *>(malloc(prefixBytes));
    if (result.prefixData) {
      memset(result.prefixData, 0, prefixBytes);
      result.prefixBytes = prefixBytes;
    }
  }

  // Allocate suffix buffer
  int32_t suffixBytes = (suffixBits + 7) / 8;
  if (suffixBytes > 0) {
    result.suffixData = static_cast<uint8_t *>(malloc(suffixBytes));
    if (result.suffixData) {
      memset(result.suffixData, 0, suffixBytes);
      result.suffixBytes = suffixBytes;
    }
  }

  // Calculate middle elements
  int64_t middleElems = (middleBits + elemWidth - 1) / elemWidth;
  int32_t middleElemBytes = (elemWidth + 7) / 8;
  if (middleElems > 0) {
    result.middle.data = malloc(middleElems * middleElemBytes);
    if (result.middle.data) {
      memset(result.middle.data, 0, middleElems * middleElemBytes);
      result.middle.len = middleElems;
    }
  }

  // Helper lambda to copy bits
  auto copyBits = [](uint8_t *dst, int64_t dstBitOffset, const uint8_t *src,
                     int64_t srcBitOffset, int64_t numBits) {
    for (int64_t i = 0; i < numBits; ++i) {
      int64_t srcByteIdx = (srcBitOffset + i) / 8;
      int64_t srcBitIdx = (srcBitOffset + i) % 8;
      int64_t dstByteIdx = (dstBitOffset + i) / 8;
      int64_t dstBitIdx = (dstBitOffset + i) % 8;

      uint8_t bit = (src[srcByteIdx] >> srcBitIdx) & 1;
      dst[dstByteIdx] |= (bit << dstBitIdx);
    }
  };

  auto *middleData = static_cast<uint8_t *>(result.middle.data);

  if (isRightToLeft) {
    // Right-to-left: suffix is at the beginning, then middle (reversed), then
    // prefix
    int64_t srcBitPos = 0;

    // Extract suffix from the beginning
    if (suffixBits > 0 && result.suffixData) {
      copyBits(result.suffixData, 0, srcData, srcBitPos, suffixBits);
      srcBitPos += suffixBits;
    }

    // Extract middle elements (reversed)
    if (middleElems > 0 && middleData) {
      for (int64_t i = middleElems - 1; i >= 0; --i) {
        int64_t bitsToExtract =
            (i == 0) ? (middleBits - (middleElems - 1) * elemWidth) : elemWidth;
        copyBits(middleData + i * middleElemBytes, 0, srcData, srcBitPos,
                 bitsToExtract);
        srcBitPos += bitsToExtract;
      }
    }

    // Extract prefix from the end
    if (prefixBits > 0 && result.prefixData) {
      copyBits(result.prefixData, 0, srcData, srcBitPos, prefixBits);
    }
  } else {
    // Left-to-right: prefix is at the beginning, then middle, then suffix
    int64_t srcBitPos = 0;

    // Extract prefix from the beginning
    if (prefixBits > 0 && result.prefixData) {
      copyBits(result.prefixData, 0, srcData, srcBitPos, prefixBits);
      srcBitPos += prefixBits;
    }

    // Extract middle elements
    if (middleElems > 0 && middleData) {
      for (int64_t i = 0; i < middleElems; ++i) {
        int64_t bitsToExtract = (i == middleElems - 1)
                                    ? (middleBits - i * elemWidth)
                                    : elemWidth;
        copyBits(middleData + i * middleElemBytes, 0, srcData, srcBitPos,
                 bitsToExtract);
        srcBitPos += bitsToExtract;
      }
    }

    // Extract suffix from the end
    if (suffixBits > 0 && result.suffixData) {
      copyBits(result.suffixData, 0, srcData, srcBitPos, suffixBits);
    }
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Event Operations
//===----------------------------------------------------------------------===//

extern "C" void __moore_event_trigger(bool *event) {
  // Trigger the event by setting its flag to true.
  // In SystemVerilog, ->event triggers the event for the current time slot.
  // Processes waiting on this event will be activated.
  if (event)
    *event = true;
}

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
// Mailbox Operations (Stubs)
//===----------------------------------------------------------------------===//
//
// These are stub implementations for the mailbox runtime functions.
// The actual implementation is in SyncPrimitivesManager, accessed via
// DPI hooks in the interpreter. These stubs exist for:
// 1. Link compatibility when compiling SystemVerilog to native code
// 2. Documentation of the expected function signatures
//

extern "C" int64_t __moore_mailbox_create(int32_t bound) {
  // Stub: In compiled simulation, this would allocate a real mailbox.
  // The interpreter handles this via SyncPrimitivesManager.
  (void)bound;
  return 0; // Invalid mailbox ID
}

extern "C" bool __moore_mailbox_tryput(int64_t mbox_id, int64_t msg) {
  // Stub: Non-blocking put - returns false (mailbox full/invalid)
  (void)mbox_id;
  (void)msg;
  return false;
}

extern "C" bool __moore_mailbox_tryget(int64_t mbox_id, int64_t *msg_out) {
  // Stub: Non-blocking get - returns false (mailbox empty/invalid)
  (void)mbox_id;
  (void)msg_out;
  return false;
}

extern "C" int64_t __moore_mailbox_num(int64_t mbox_id) {
  // Stub: Returns 0 (empty or invalid mailbox)
  (void)mbox_id;
  return 0;
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
// Randomization Operations
//===----------------------------------------------------------------------===//

extern "C" int32_t __moore_randomize_basic(void *classPtr, int64_t classSize) {
  // Validate inputs
  if (!classPtr || classSize <= 0)
    return 0;

  // Fill the class memory with random values using __moore_urandom.
  // Process in 4-byte chunks for efficiency, then handle remaining bytes.
  auto *data = static_cast<uint8_t *>(classPtr);
  int64_t fullWords = classSize / 4;
  int64_t remainingBytes = classSize % 4;

  // Fill 4-byte words
  auto *wordPtr = reinterpret_cast<uint32_t *>(data);
  for (int64_t i = 0; i < fullWords; ++i) {
    wordPtr[i] = __moore_urandom();
  }

  // Fill remaining bytes (if any)
  if (remainingBytes > 0) {
    uint32_t lastWord = __moore_urandom();
    uint8_t *remainingPtr = data + fullWords * 4;
    for (int64_t i = 0; i < remainingBytes; ++i) {
      remainingPtr[i] = static_cast<uint8_t>((lastWord >> (i * 8)) & 0xFF);
    }
  }

  return 1; // Success
}

extern "C" int64_t __moore_randc_next(void *fieldPtr, int64_t bitWidth) {
  if (!fieldPtr || bitWidth <= 0)
    return 0;

  struct RandCState {
    int64_t bitWidth = 0;
    std::vector<uint64_t> remaining;
    uint64_t current = 0;
    uint64_t step = 1;
    bool linear = false;
    bool initialized = false;
  };

  static std::mutex randcMutex;
  static std::unordered_map<void *, RandCState> randcStates;

  std::lock_guard<std::mutex> lock(randcMutex);
  auto &state = randcStates[fieldPtr];
  if (bitWidth > 63) {
    uint64_t value = static_cast<uint64_t>(__moore_urandom());
    value |= static_cast<uint64_t>(__moore_urandom()) << 32;
    return static_cast<int64_t>(value);
  }

  constexpr int64_t kMaxCycleBits = 16;
  const uint64_t maxValue = (1ULL << bitWidth) - 1;

  if (state.bitWidth != bitWidth) {
    state.bitWidth = bitWidth;
    state.remaining.clear();
    state.linear = bitWidth > kMaxCycleBits;
    state.initialized = false;
  }

  if (!state.linear) {
    if (state.remaining.empty()) {
      state.remaining.reserve(maxValue + 1);
      for (uint64_t value = 0; value <= maxValue; ++value)
        state.remaining.push_back(value);
    }

    std::uniform_int_distribution<size_t> dist(0, state.remaining.size() - 1);
    size_t idx = dist(urandomGenerator);
    uint64_t value = state.remaining[idx];
    state.remaining[idx] = state.remaining.back();
    state.remaining.pop_back();
    return static_cast<int64_t>(value);
  }

  if (!state.initialized) {
    uint64_t seed = static_cast<uint64_t>(__moore_urandom());
    seed |= static_cast<uint64_t>(__moore_urandom()) << 32;
    state.current = seed & maxValue;
    uint64_t step = seed | 1ULL;
    step &= maxValue;
    if (step == 0)
      step = 1;
    state.step = step;
    state.initialized = true;
    return static_cast<int64_t>(state.current);
  }

  state.current = (state.current + state.step) & maxValue;
  return static_cast<int64_t>(state.current);
}

extern "C" int64_t __moore_randomize_with_dist(int64_t *ranges, int64_t *weights,
                                               int64_t *perRange,
                                               int64_t numRanges) {
  if (!ranges || !weights || !perRange || numRanges <= 0)
    return 0;

  // Calculate total weight considering per-range vs per-value weights.
  // For := (perRange=0), weight applies to each value in the range.
  // For :/ (perRange=1), weight is divided among values in the range.
  int64_t totalWeight = 0;
  std::vector<int64_t> effectiveWeights(numRanges);

  for (int64_t i = 0; i < numRanges; ++i) {
    int64_t low = ranges[i * 2];
    int64_t high = ranges[i * 2 + 1];
    int64_t rangeSize = high - low + 1;
    int64_t weight = weights[i];

    if (perRange[i] == 0) {
      // := weight: weight applies to each value
      effectiveWeights[i] = weight * rangeSize;
    } else {
      // :/ weight: total weight for the range
      effectiveWeights[i] = weight;
    }
    totalWeight += effectiveWeights[i];
  }

  if (totalWeight <= 0)
    return ranges[0]; // Return first value if no valid weights

  // Generate a random number in [0, totalWeight)
  int64_t randomWeight =
      static_cast<int64_t>(__moore_urandom()) % totalWeight;

  // Find which range the random weight falls into
  int64_t cumulativeWeight = 0;
  for (int64_t i = 0; i < numRanges; ++i) {
    cumulativeWeight += effectiveWeights[i];
    if (randomWeight < cumulativeWeight) {
      // Select from this range
      int64_t low = ranges[i * 2];
      int64_t high = ranges[i * 2 + 1];
      int64_t rangeSize = high - low + 1;

      // Pick a random value within the range
      if (rangeSize == 1) {
        return low;
      } else {
        return low + static_cast<int64_t>(__moore_urandom()) % rangeSize;
      }
    }
  }

  // Fallback: return first value (shouldn't reach here)
  return ranges[0];
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
// Array Locator Methods
//===----------------------------------------------------------------------===//

namespace {

/// Helper to read an element value as a 64-bit integer.
/// Handles elements up to 8 bytes (64 bits).
int64_t readElementValue(void *element, int64_t elementSize) {
  int64_t value = 0;
  if (elementSize > 0 && elementSize <= 8) {
    std::memcpy(&value, element,
                static_cast<size_t>(elementSize));
  }
  return value;
}

/// Helper to read an element value as an unsigned 64-bit integer.
uint64_t readElementValueUnsigned(void *element, int64_t elementSize) {
  uint64_t value = 0;
  if (elementSize > 0 && elementSize <= 8) {
    std::memcpy(&value, element,
                static_cast<size_t>(elementSize));
  }
  return value;
}

/// Helper to add an element to a result queue.
/// Returns the new queue with the element appended.
MooreQueue appendToResult(MooreQueue result, void *element,
                          int64_t elementSize) {
  int64_t newLen = result.len + 1;
  void *newData = std::realloc(result.data, newLen * elementSize);
  if (!newData) {
    // Allocation failed, return unchanged
    return result;
  }
  std::memcpy(static_cast<char *>(newData) + result.len * elementSize,
              element, elementSize);
  result.data = newData;
  result.len = newLen;
  return result;
}

/// Helper to add an index to a result queue (indices are stored as int64_t).
MooreQueue appendIndexToResult(MooreQueue result, int64_t index) {
  int64_t newLen = result.len + 1;
  void *newData = std::realloc(result.data, newLen * sizeof(int64_t));
  if (!newData) {
    return result;
  }
  static_cast<int64_t *>(newData)[result.len] = index;
  result.data = newData;
  result.len = newLen;
  return result;
}

} // anonymous namespace

extern "C" MooreQueue __moore_array_locator(MooreQueue *array,
                                            int64_t elementSize,
                                            MooreLocatorPredicate predicate,
                                            void *userData, int32_t mode,
                                            bool returnIndices) {
  MooreQueue result = {nullptr, 0};

  // Validate inputs
  if (!array || !array->data || array->len <= 0 || elementSize <= 0 ||
      !predicate) {
    return result;
  }

  auto *data = static_cast<char *>(array->data);
  int64_t numElements = array->len;

  // Mode: 0=all, 1=first, 2=last
  if (mode == 2) {
    // Last: iterate backwards
    for (int64_t i = numElements - 1; i >= 0; --i) {
      void *element = data + i * elementSize;
      if (predicate(element, userData)) {
        if (returnIndices) {
          result = appendIndexToResult(result, i);
        } else {
          result = appendToResult(result, element, elementSize);
        }
        // For "last", we only want one result
        break;
      }
    }
  } else if (mode == 1) {
    // First: iterate forwards, stop at first match
    for (int64_t i = 0; i < numElements; ++i) {
      void *element = data + i * elementSize;
      if (predicate(element, userData)) {
        if (returnIndices) {
          result = appendIndexToResult(result, i);
        } else {
          result = appendToResult(result, element, elementSize);
        }
        break;
      }
    }
  } else {
    // All: iterate forwards, collect all matches
    for (int64_t i = 0; i < numElements; ++i) {
      void *element = data + i * elementSize;
      if (predicate(element, userData)) {
        if (returnIndices) {
          result = appendIndexToResult(result, i);
        } else {
          result = appendToResult(result, element, elementSize);
        }
      }
    }
  }

  return result;
}

extern "C" MooreQueue __moore_array_find_eq(MooreQueue *array,
                                            int64_t elementSize, void *value,
                                            int32_t mode, bool returnIndices) {
  MooreQueue result = {nullptr, 0};

  // Validate inputs
  if (!array || !array->data || array->len <= 0 || elementSize <= 0 ||
      !value) {
    return result;
  }

  auto *data = static_cast<char *>(array->data);
  int64_t numElements = array->len;

  // Mode: 0=all, 1=first, 2=last
  if (mode == 2) {
    // Last: iterate backwards
    for (int64_t i = numElements - 1; i >= 0; --i) {
      void *element = data + i * elementSize;
      if (std::memcmp(element, value, elementSize) == 0) {
        if (returnIndices) {
          result = appendIndexToResult(result, i);
        } else {
          result = appendToResult(result, element, elementSize);
        }
        break;
      }
    }
  } else if (mode == 1) {
    // First: iterate forwards, stop at first match
    for (int64_t i = 0; i < numElements; ++i) {
      void *element = data + i * elementSize;
      if (std::memcmp(element, value, elementSize) == 0) {
        if (returnIndices) {
          result = appendIndexToResult(result, i);
        } else {
          result = appendToResult(result, element, elementSize);
        }
        break;
      }
    }
  } else {
    // All: iterate forwards, collect all matches
    for (int64_t i = 0; i < numElements; ++i) {
      void *element = data + i * elementSize;
      if (std::memcmp(element, value, elementSize) == 0) {
        if (returnIndices) {
          result = appendIndexToResult(result, i);
        } else {
          result = appendToResult(result, element, elementSize);
        }
      }
    }
  }

  return result;
}

extern "C" MooreQueue __moore_array_find_cmp(MooreQueue *array,
                                             int64_t elementSize, void *value,
                                             int32_t cmpMode,
                                             int32_t locatorMode,
                                             bool returnIndices) {
  MooreQueue result = {nullptr, 0};

  // Validate inputs
  if (!array || !array->data || array->len <= 0 || elementSize <= 0 ||
      !value || elementSize > 8) {
    return result;
  }

  auto *data = static_cast<char *>(array->data);
  int64_t numElements = array->len;

  // Read the comparison value as a signed 64-bit integer
  int64_t cmpValue = readElementValue(value, elementSize);
  // Sign-extend if needed
  if (elementSize < 8) {
    int shift = (8 - elementSize) * 8;
    cmpValue = (cmpValue << shift) >> shift;
  }

  // Helper lambda to check if an element matches the comparison predicate
  auto matchesPredicate = [&](void *element) -> bool {
    int64_t elemValue = readElementValue(element, elementSize);
    // Sign-extend if needed
    if (elementSize < 8) {
      int shift = (8 - elementSize) * 8;
      elemValue = (elemValue << shift) >> shift;
    }

    switch (cmpMode) {
    case MOORE_CMP_EQ:  // Equal
      return elemValue == cmpValue;
    case MOORE_CMP_NE:  // Not equal
      return elemValue != cmpValue;
    case MOORE_CMP_SGT: // Signed greater than
      return elemValue > cmpValue;
    case MOORE_CMP_SGE: // Signed greater than or equal
      return elemValue >= cmpValue;
    case MOORE_CMP_SLT: // Signed less than
      return elemValue < cmpValue;
    case MOORE_CMP_SLE: // Signed less than or equal
      return elemValue <= cmpValue;
    default:
      return false;
    }
  };

  // locatorMode: 0=all, 1=first, 2=last
  if (locatorMode == 2) {
    // Last: iterate backwards
    for (int64_t i = numElements - 1; i >= 0; --i) {
      void *element = data + i * elementSize;
      if (matchesPredicate(element)) {
        if (returnIndices) {
          result = appendIndexToResult(result, i);
        } else {
          result = appendToResult(result, element, elementSize);
        }
        break;
      }
    }
  } else if (locatorMode == 1) {
    // First: iterate forwards, stop at first match
    for (int64_t i = 0; i < numElements; ++i) {
      void *element = data + i * elementSize;
      if (matchesPredicate(element)) {
        if (returnIndices) {
          result = appendIndexToResult(result, i);
        } else {
          result = appendToResult(result, element, elementSize);
        }
        break;
      }
    }
  } else {
    // All: iterate forwards, collect all matches
    for (int64_t i = 0; i < numElements; ++i) {
      void *element = data + i * elementSize;
      if (matchesPredicate(element)) {
        if (returnIndices) {
          result = appendIndexToResult(result, i);
        } else {
          result = appendToResult(result, element, elementSize);
        }
      }
    }
  }

  return result;
}

extern "C" MooreQueue __moore_array_find_field_cmp(MooreQueue *array,
                                                   int64_t elementSize,
                                                   int64_t fieldOffset,
                                                   int64_t fieldSize,
                                                   void *value, int32_t cmpMode,
                                                   int32_t locatorMode,
                                                   bool returnIndices) {
  MooreQueue result = {nullptr, 0};

  // Validate inputs
  // Note: elementSize is the size of the class handle (pointer), typically 8 bytes.
  // fieldOffset is the offset within the object pointed to by the class handle.
  // We don't validate fieldOffset against elementSize since they refer to
  // different things (pointer size vs object layout).
  if (!array || !array->data || array->len <= 0 || elementSize <= 0 ||
      !value || fieldSize <= 0 || fieldSize > 8 || fieldOffset < 0) {
    return result;
  }

  auto *data = static_cast<char *>(array->data);
  int64_t numElements = array->len;

  // Read the comparison value as a signed 64-bit integer
  int64_t cmpValue = readElementValue(value, fieldSize);
  // Sign-extend if needed
  if (fieldSize < 8) {
    int shift = (8 - fieldSize) * 8;
    cmpValue = (cmpValue << shift) >> shift;
  }

  // Helper lambda to check if an element's field matches the comparison predicate.
  // For class handles, the element is a pointer to an object. We need to:
  // 1. Read the pointer value from the array element
  // 2. Dereference to get the object
  // 3. Access the field at the specified offset within the object
  auto matchesPredicate = [&](void *elementPtr) -> bool {
    // elementPtr points to the class handle (pointer) stored in the array.
    // Read the pointer value to get the object address.
    void *objectPtr = *static_cast<void **>(elementPtr);
    if (!objectPtr)
      return false; // Null class handle

    // Access the field at the specified offset within the object
    void *fieldPtr = static_cast<char *>(objectPtr) + fieldOffset;
    int64_t fieldValue = readElementValue(fieldPtr, fieldSize);
    // Sign-extend if needed
    if (fieldSize < 8) {
      int shift = (8 - fieldSize) * 8;
      fieldValue = (fieldValue << shift) >> shift;
    }

    switch (cmpMode) {
    case MOORE_CMP_EQ:  // Equal
      return fieldValue == cmpValue;
    case MOORE_CMP_NE:  // Not equal
      return fieldValue != cmpValue;
    case MOORE_CMP_SGT: // Signed greater than
      return fieldValue > cmpValue;
    case MOORE_CMP_SGE: // Signed greater than or equal
      return fieldValue >= cmpValue;
    case MOORE_CMP_SLT: // Signed less than
      return fieldValue < cmpValue;
    case MOORE_CMP_SLE: // Signed less than or equal
      return fieldValue <= cmpValue;
    default:
      return false;
    }
  };

  // locatorMode: 0=all, 1=first, 2=last
  if (locatorMode == 2) {
    // Last: iterate backwards
    for (int64_t i = numElements - 1; i >= 0; --i) {
      void *element = data + i * elementSize;
      if (matchesPredicate(element)) {
        if (returnIndices) {
          result = appendIndexToResult(result, i);
        } else {
          result = appendToResult(result, element, elementSize);
        }
        break;
      }
    }
  } else if (locatorMode == 1) {
    // First: iterate forwards, stop at first match
    for (int64_t i = 0; i < numElements; ++i) {
      void *element = data + i * elementSize;
      if (matchesPredicate(element)) {
        if (returnIndices) {
          result = appendIndexToResult(result, i);
        } else {
          result = appendToResult(result, element, elementSize);
        }
        break;
      }
    }
  } else {
    // All: iterate forwards, collect all matches
    for (int64_t i = 0; i < numElements; ++i) {
      void *element = data + i * elementSize;
      if (matchesPredicate(element)) {
        if (returnIndices) {
          result = appendIndexToResult(result, i);
        } else {
          result = appendToResult(result, element, elementSize);
        }
      }
    }
  }

  return result;
}

extern "C" MooreQueue __moore_array_min(MooreQueue *array, int64_t elementSize,
                                        bool isSigned) {
  MooreQueue result = {nullptr, 0};

  // Validate inputs
  if (!array || !array->data || array->len <= 0 || elementSize <= 0 ||
      elementSize > 8) {
    return result;
  }

  auto *data = static_cast<char *>(array->data);
  int64_t numElements = array->len;

  // Find the minimum value
  int64_t minIndex = 0;
  if (isSigned) {
    int64_t minValue = readElementValue(data, elementSize);
    // Sign-extend if needed
    if (elementSize < 8) {
      int shift = (8 - elementSize) * 8;
      minValue = (minValue << shift) >> shift;
    }
    for (int64_t i = 1; i < numElements; ++i) {
      int64_t value = readElementValue(data + i * elementSize, elementSize);
      if (elementSize < 8) {
        int shift = (8 - elementSize) * 8;
        value = (value << shift) >> shift;
      }
      if (value < minValue) {
        minValue = value;
        minIndex = i;
      }
    }
  } else {
    uint64_t minValue = readElementValueUnsigned(data, elementSize);
    for (int64_t i = 1; i < numElements; ++i) {
      uint64_t value =
          readElementValueUnsigned(data + i * elementSize, elementSize);
      if (value < minValue) {
        minValue = value;
        minIndex = i;
      }
    }
  }

  // Return the minimum element
  result = appendToResult(result, data + minIndex * elementSize, elementSize);
  return result;
}

extern "C" MooreQueue __moore_array_max(MooreQueue *array, int64_t elementSize,
                                        bool isSigned) {
  MooreQueue result = {nullptr, 0};

  // Validate inputs
  if (!array || !array->data || array->len <= 0 || elementSize <= 0 ||
      elementSize > 8) {
    return result;
  }

  auto *data = static_cast<char *>(array->data);
  int64_t numElements = array->len;

  // Find the maximum value
  int64_t maxIndex = 0;
  if (isSigned) {
    int64_t maxValue = readElementValue(data, elementSize);
    // Sign-extend if needed
    if (elementSize < 8) {
      int shift = (8 - elementSize) * 8;
      maxValue = (maxValue << shift) >> shift;
    }
    for (int64_t i = 1; i < numElements; ++i) {
      int64_t value = readElementValue(data + i * elementSize, elementSize);
      if (elementSize < 8) {
        int shift = (8 - elementSize) * 8;
        value = (value << shift) >> shift;
      }
      if (value > maxValue) {
        maxValue = value;
        maxIndex = i;
      }
    }
  } else {
    uint64_t maxValue = readElementValueUnsigned(data, elementSize);
    for (int64_t i = 1; i < numElements; ++i) {
      uint64_t value =
          readElementValueUnsigned(data + i * elementSize, elementSize);
      if (value > maxValue) {
        maxValue = value;
        maxIndex = i;
      }
    }
  }

  // Return the maximum element
  result = appendToResult(result, data + maxIndex * elementSize, elementSize);
  return result;
}

extern "C" MooreQueue __moore_array_unique(MooreQueue *array,
                                           int64_t elementSize) {
  MooreQueue result = {nullptr, 0};

  // Validate inputs
  if (!array || !array->data || array->len <= 0 || elementSize <= 0) {
    return result;
  }

  auto *data = static_cast<char *>(array->data);
  int64_t numElements = array->len;

  // For each element, check if it's already in the result
  for (int64_t i = 0; i < numElements; ++i) {
    void *element = data + i * elementSize;
    bool found = false;

    // Check if this element is already in the result
    auto *resultData = static_cast<char *>(result.data);
    for (int64_t j = 0; j < result.len; ++j) {
      if (std::memcmp(resultData + j * elementSize, element, elementSize) ==
          0) {
        found = true;
        break;
      }
    }

    if (!found) {
      result = appendToResult(result, element, elementSize);
    }
  }

  return result;
}

extern "C" MooreQueue __moore_array_unique_index(MooreQueue *array,
                                                 int64_t elementSize) {
  MooreQueue result = {nullptr, 0};

  // Validate inputs
  if (!array || !array->data || array->len <= 0 || elementSize <= 0) {
    return result;
  }

  auto *data = static_cast<char *>(array->data);
  int64_t numElements = array->len;

  // Track which unique values we've seen (using a simple list approach)
  // For better performance with large arrays, a hash set would be preferred
  MooreQueue seenElements = {nullptr, 0};

  for (int64_t i = 0; i < numElements; ++i) {
    void *element = data + i * elementSize;
    bool found = false;

    // Check if this element value has been seen before
    auto *seenData = static_cast<char *>(seenElements.data);
    for (int64_t j = 0; j < seenElements.len; ++j) {
      if (std::memcmp(seenData + j * elementSize, element, elementSize) == 0) {
        found = true;
        break;
      }
    }

    if (!found) {
      // Add the index to result and the element to seen list
      result = appendIndexToResult(result, i);
      seenElements = appendToResult(seenElements, element, elementSize);
    }
  }

  // Free the temporary seen elements storage
  if (seenElements.data) {
    std::free(seenElements.data);
  }

  return result;
}

extern "C" int64_t __moore_array_reduce_sum(MooreQueue *array,
                                            int64_t elementSize) {
  if (!array || !array->data || array->len <= 0 || elementSize <= 0 ||
      elementSize > 8)
    return 0;

  auto *data = static_cast<char *>(array->data);
  int64_t numElements = array->len;

  uint64_t acc = 0;
  for (int64_t i = 0; i < numElements; ++i) {
    uint64_t value = readElementValueUnsigned(data + i * elementSize,
                                              elementSize);
    acc += value;
  }
  return static_cast<int64_t>(acc);
}

extern "C" int64_t __moore_array_reduce_product(MooreQueue *array,
                                                int64_t elementSize) {
  if (!array || !array->data || array->len <= 0 || elementSize <= 0 ||
      elementSize > 8)
    return 1;

  auto *data = static_cast<char *>(array->data);
  int64_t numElements = array->len;

  uint64_t acc = 1;
  for (int64_t i = 0; i < numElements; ++i) {
    uint64_t value = readElementValueUnsigned(data + i * elementSize,
                                              elementSize);
    acc *= value;
  }
  return static_cast<int64_t>(acc);
}

extern "C" int64_t __moore_array_reduce_and(MooreQueue *array,
                                            int64_t elementSize) {
  if (!array || !array->data || array->len <= 0 || elementSize <= 0 ||
      elementSize > 8)
    return 0;

  auto *data = static_cast<char *>(array->data);
  int64_t numElements = array->len;

  uint64_t mask = elementSize >= 8
                      ? ~static_cast<uint64_t>(0)
                      : ((static_cast<uint64_t>(1) << (elementSize * 8)) - 1);
  uint64_t acc = mask;
  for (int64_t i = 0; i < numElements; ++i) {
    uint64_t value = readElementValueUnsigned(data + i * elementSize,
                                              elementSize);
    acc &= value;
  }
  return static_cast<int64_t>(acc);
}

extern "C" int64_t __moore_array_reduce_or(MooreQueue *array,
                                           int64_t elementSize) {
  if (!array || !array->data || array->len <= 0 || elementSize <= 0 ||
      elementSize > 8)
    return 0;

  auto *data = static_cast<char *>(array->data);
  int64_t numElements = array->len;

  uint64_t acc = 0;
  for (int64_t i = 0; i < numElements; ++i) {
    uint64_t value = readElementValueUnsigned(data + i * elementSize,
                                              elementSize);
    acc |= value;
  }
  return static_cast<int64_t>(acc);
}

extern "C" int64_t __moore_array_reduce_xor(MooreQueue *array,
                                            int64_t elementSize) {
  if (!array || !array->data || array->len <= 0 || elementSize <= 0 ||
      elementSize > 8)
    return 0;

  auto *data = static_cast<char *>(array->data);
  int64_t numElements = array->len;

  uint64_t acc = 0;
  for (int64_t i = 0; i < numElements; ++i) {
    uint64_t value = readElementValueUnsigned(data + i * elementSize,
                                              elementSize);
    acc ^= value;
  }
  return static_cast<int64_t>(acc);
}

//===----------------------------------------------------------------------===//
// Coverage Collection Operations
//===----------------------------------------------------------------------===//
//
// These functions provide runtime support for SystemVerilog coverage
// collection. Covergroups track which values have been observed during
// simulation, enabling functional coverage analysis.
//

namespace {

/// Global list of all registered covergroups for reporting.
/// Thread-local to avoid synchronization issues in multi-threaded simulations.
thread_local std::vector<MooreCovergroup *> registeredCovergroups;

/// Global test name for coverage database operations.
/// This is used when saving coverage databases to identify the test run.
thread_local std::string globalTestName;

/// Helper to track unique values seen by a coverpoint using a simple set.
/// For production use, this could be replaced with a more efficient data
/// structure like a hash set or bit vector.
struct CoverpointTracker {
  std::map<int64_t, int64_t> valueCounts; // value -> hit count

  // Previous value tracking for transition bins
  int64_t prevValue;
  bool hasPrevValue;

  CoverpointTracker() : prevValue(0), hasPrevValue(false) {}
};

/// Map from coverpoint to its value tracker.
/// This is separate from the MooreCoverpoint struct to keep the C API simple.
thread_local std::map<MooreCoverpoint *, CoverpointTracker> coverpointTrackers;

/// Coverage options for covergroups (IEEE 1800-2017 Section 19.7.1).
struct CovergroupOptions {
  int64_t weight = 1;         ///< option.weight (default: 1)
  bool perInstance = false;   ///< option.per_instance (default: false)
  int64_t atLeast = 1;        ///< option.at_least (default: 1)
  int64_t autoBinMax = 64;    ///< option.auto_bin_max (default: 64)
  std::string comment;        ///< option.comment (default: empty)
};

/// Map from covergroup to its options.
thread_local std::map<MooreCovergroup *, CovergroupOptions> covergroupOptions;

/// Coverage options for coverpoints (IEEE 1800-2017 Section 19.7.2).
struct CoverpointOptions {
  int64_t weight = 1;       ///< option.weight (default: 1)
  double goal = 100.0;      ///< option.goal (default: 100.0)
  int64_t atLeast = 1;      ///< option.at_least (default: 1)
  int64_t autoBinMax = 64;  ///< option.auto_bin_max (default: 64)
  std::string comment;      ///< option.comment (default: empty)
};

/// Map from coverpoint to its options.
thread_local std::map<MooreCoverpoint *, CoverpointOptions> coverpointOptions;

/// Structure to store a transition bin with its sequences and state.
struct TransitionBin {
  const char *name;
  std::vector<MooreTransitionSequence> sequences; // Alternative sequences
  int64_t hit_count;

  // State for tracking sequence progress per alternative
  struct SequenceState {
    int32_t currentStep;  // Which step in the sequence we're waiting for
    int32_t repeatCount;  // For repeat patterns, count of matching values
    bool active;          // Is this sequence currently being tracked

    SequenceState() : currentStep(0), repeatCount(0), active(false) {}
  };
  std::vector<SequenceState> sequenceStates;

  TransitionBin() : name(nullptr), hit_count(0) {}
};

/// Structure to store explicit bin data for a coverpoint.
/// Stored separately to maintain backward compatibility with the C API.
struct ExplicitBinData {
  std::vector<MooreCoverageBin> bins;
  std::vector<TransitionBin> transitionBins;
};

/// Map from coverpoint to its explicit bin data.
thread_local std::map<MooreCoverpoint *, ExplicitBinData> explicitBinData;

/// Set of excluded bin names for each coverpoint.
/// Excluded bins are not counted toward coverage goals.
/// This supports IEEE 1800-2017 coverage exclusion semantics.
thread_local std::map<MooreCoverpoint *, std::set<std::string>> excludedBins;

/// Global exclusion file path for bulk exclusion loading.
thread_local std::string globalExclusionFile;

} // anonymous namespace

extern "C" void *__moore_covergroup_create(const char *name,
                                            int32_t num_coverpoints) {
  // Validate inputs
  if (num_coverpoints < 0)
    return nullptr;

  // Allocate the covergroup structure
  auto *cg = static_cast<MooreCovergroup *>(std::malloc(sizeof(MooreCovergroup)));
  if (!cg)
    return nullptr;

  // Initialize the covergroup - make a copy of the name to avoid dangling pointers
  cg->name = name ? strdup(name) : nullptr;
  cg->num_coverpoints = num_coverpoints;

  // Allocate the coverpoints array
  if (num_coverpoints > 0) {
    cg->coverpoints = static_cast<MooreCoverpoint **>(
        std::calloc(num_coverpoints, sizeof(MooreCoverpoint *)));
    if (!cg->coverpoints) {
      std::free(cg);
      return nullptr;
    }
  } else {
    cg->coverpoints = nullptr;
  }

  // Register this covergroup for reporting
  registeredCovergroups.push_back(cg);

  return cg;
}

extern "C" void __moore_coverpoint_init(void *cg, int32_t cp_index,
                                         const char *name) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;

  // Allocate a new coverpoint
  auto *cp = static_cast<MooreCoverpoint *>(std::malloc(sizeof(MooreCoverpoint)));
  if (!cp)
    return;

  // Initialize the coverpoint with auto bins (no explicit bins)
  // Make a copy of the name to avoid dangling pointers
  cp->name = name ? strdup(name) : nullptr;
  cp->bins = nullptr;
  cp->num_bins = 0;
  cp->hits = 0;
  cp->min_val = INT64_MAX; // Will be updated on first sample
  cp->max_val = INT64_MIN; // Will be updated on first sample

  // Store the coverpoint in the covergroup
  covergroup->coverpoints[cp_index] = cp;

  // Initialize the value tracker for this coverpoint
  coverpointTrackers[cp] = CoverpointTracker();
}

extern "C" void __moore_covergroup_destroy(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;

  // Remove from registered list
  auto it = std::find(registeredCovergroups.begin(),
                      registeredCovergroups.end(), covergroup);
  if (it != registeredCovergroups.end()) {
    registeredCovergroups.erase(it);
  }

  // Free each coverpoint
  for (int32_t i = 0; i < covergroup->num_coverpoints; ++i) {
    if (covergroup->coverpoints[i]) {
      // Remove from tracker map
      coverpointTrackers.erase(covergroup->coverpoints[i]);

      // Remove from options map
      coverpointOptions.erase(covergroup->coverpoints[i]);

      // Remove from explicit bin data map
      explicitBinData.erase(covergroup->coverpoints[i]);

      // Remove from excluded bins map
      excludedBins.erase(covergroup->coverpoints[i]);

      // Free bin array if present
      if (covergroup->coverpoints[i]->bins) {
        std::free(covergroup->coverpoints[i]->bins);
      }
      // Free coverpoint name (allocated by strdup)
      if (covergroup->coverpoints[i]->name) {
        std::free(const_cast<char *>(covergroup->coverpoints[i]->name));
      }
      std::free(covergroup->coverpoints[i]);
    }
  }

  // Free the coverpoints array
  if (covergroup->coverpoints) {
    std::free(covergroup->coverpoints);
  }

  // Clean up cross coverage data (implemented later in file)
  extern void __moore_cross_cleanup_for_covergroup(MooreCovergroup *);
  __moore_cross_cleanup_for_covergroup(covergroup);

  // Clean up sample callbacks (implemented later in file)
  extern void __moore_sample_callbacks_cleanup_for_covergroup(MooreCovergroup *);
  __moore_sample_callbacks_cleanup_for_covergroup(covergroup);

  // Clean up covergroup options
  covergroupOptions.erase(covergroup);

  // Free the covergroup name (allocated by strdup)
  if (covergroup->name) {
    std::free(const_cast<char *>(covergroup->name));
  }

  // Free the covergroup itself
  std::free(covergroup);
}

// Forward declaration for explicit bin update helper
namespace {
void updateExplicitBinsHelper(MooreCoverpoint *cp, int64_t value);
void updateTransitionBinsHelper(MooreCoverpoint *cp, int64_t value,
                                int64_t prevValue, bool hasPrev);
// Forward declarations for illegal/ignore bin checking
bool checkIllegalBinsInternal(MooreCovergroup *cg, MooreCoverpoint *cp,
                              int64_t value);
bool checkIgnoreBinsInternal(MooreCoverpoint *cp, int64_t value);
} // namespace

extern "C" void __moore_coverpoint_sample(void *cg, int32_t cp_index,
                                           int64_t value) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return;

  // Check for ignore bins first - skip sampling if matched
  if (checkIgnoreBinsInternal(cp, value)) {
    // Value is in an ignore bin, don't count it
    return;
  }

  // Check for illegal bins - report error/warning but continue
  checkIllegalBinsInternal(covergroup, cp, value);

  // Update hit count
  cp->hits++;

  // Update min/max tracking
  if (value < cp->min_val)
    cp->min_val = value;
  if (value > cp->max_val)
    cp->max_val = value;

  // Track unique values and get previous value for transition tracking
  // Create tracker if it doesn't exist (defensive - should already exist from init)
  auto &tracker = coverpointTrackers[cp];
  int64_t prevValue = tracker.prevValue;
  bool hasPrev = tracker.hasPrevValue;
  tracker.valueCounts[value]++;

  // Update explicit bins if present
  updateExplicitBinsHelper(cp, value);

  // Update transition bins if present
  updateTransitionBinsHelper(cp, value, prevValue, hasPrev);

  // Update previous value for transition tracking
  tracker.prevValue = value;
  tracker.hasPrevValue = true;
}

extern "C" double __moore_coverpoint_get_coverage(void *cg, int32_t cp_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return 0.0;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp || cp->hits == 0)
    return 0.0;

  // Get the at_least threshold for this coverpoint
  // First check coverpoint-specific option, then fall back to covergroup option
  int64_t atLeast = 1;
  auto cpOptIt = coverpointOptions.find(cp);
  if (cpOptIt != coverpointOptions.end()) {
    atLeast = cpOptIt->second.atLeast;
  } else {
    // Fall back to covergroup-level at_least
    auto cgOptIt = covergroupOptions.find(covergroup);
    if (cgOptIt != covergroupOptions.end()) {
      atLeast = cgOptIt->second.atLeast;
    }
  }

  // For auto bins, calculate coverage as the ratio of unique values seen
  // to the theoretical range. Since we don't know the type's range at runtime,
  // we use the actual range of values seen plus some margin.
  auto trackerIt = coverpointTrackers.find(cp);
  if (trackerIt == coverpointTrackers.end())
    return 0.0;

  // If we have explicit bins, use those
  if (cp->bins && cp->num_bins > 0) {
    int64_t coveredBins = 0;
    int64_t totalBins = 0;

    // Check explicit bin data for ignore bins (which shouldn't count)
    auto binDataIt = explicitBinData.find(cp);

    // Check excluded bins set
    auto excludedIt = excludedBins.find(cp);

    for (int32_t i = 0; i < cp->num_bins; ++i) {
      // Skip ignore bins - they don't count toward coverage
      if (binDataIt != explicitBinData.end() &&
          i < static_cast<int32_t>(binDataIt->second.bins.size()) &&
          binDataIt->second.bins[i].kind == MOORE_BIN_KIND_IGNORE) {
        continue;
      }

      // Skip excluded bins - they don't count toward coverage
      if (binDataIt != explicitBinData.end() &&
          i < static_cast<int32_t>(binDataIt->second.bins.size()) &&
          excludedIt != excludedBins.end() &&
          binDataIt->second.bins[i].name != nullptr &&
          excludedIt->second.count(binDataIt->second.bins[i].name) > 0) {
        continue;
      }

      totalBins++;
      // A bin is covered if its hit count >= at_least threshold
      if (cp->bins[i] >= atLeast)
        coveredBins++;
    }

    if (totalBins == 0)
      return 100.0;  // No countable bins means 100% coverage

    return (100.0 * coveredBins) / totalBins;
  }

  // For auto bins, estimate coverage based on unique values seen.
  // Count values that have been hit at least 'atLeast' times
  int64_t coveredValues = 0;
  for (const auto &entry : trackerIt->second.valueCounts) {
    if (entry.second >= atLeast)
      coveredValues++;
  }

  // We assume the goal is to cover the range [min_val, max_val].
  // If the range is 0 (single value), coverage is 100%.
  if (cp->min_val > cp->max_val)
    return 0.0; // No valid samples

  int64_t range = cp->max_val - cp->min_val + 1;
  if (range <= 0)
    return 100.0; // Single value = 100% coverage

  // Get auto_bin_max to limit the number of bins considered
  int64_t autoBinMax = 64;
  if (cpOptIt != coverpointOptions.end()) {
    autoBinMax = cpOptIt->second.autoBinMax;
  } else {
    auto cgOptIt = covergroupOptions.find(covergroup);
    if (cgOptIt != covergroupOptions.end()) {
      autoBinMax = cgOptIt->second.autoBinMax;
    }
  }

  // If range exceeds auto_bin_max, clamp the effective range for coverage calc
  int64_t effectiveRange = std::min(range, autoBinMax);

  // Calculate coverage percentage
  // Cap at 100% since covered values might exceed expected range
  double coverage = (100.0 * coveredValues) / effectiveRange;
  return coverage > 100.0 ? 100.0 : coverage;
}

extern "C" double __moore_covergroup_get_coverage(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || covergroup->num_coverpoints == 0)
    return 0.0;

  // Check if per_instance is false - if so, aggregate across all instances
  // with the same type name
  auto cgOptIt = covergroupOptions.find(covergroup);
  bool perInstance = false;
  if (cgOptIt != covergroupOptions.end()) {
    perInstance = cgOptIt->second.perInstance;
  }

  if (!perInstance && covergroup->name) {
    // Aggregate coverage across all instances with the same name
    double totalCoverage = 0.0;
    int32_t instanceCount = 0;

    for (auto *otherCg : registeredCovergroups) {
      if (otherCg && otherCg->name &&
          std::strcmp(otherCg->name, covergroup->name) == 0) {
        totalCoverage += __moore_covergroup_get_inst_coverage(otherCg);
        instanceCount++;
      }
    }

    if (instanceCount == 0)
      return 0.0;

    return totalCoverage / instanceCount;
  }

  // per_instance mode or no name - return instance-specific coverage
  return __moore_covergroup_get_inst_coverage(cg);
}

extern "C" double __moore_covergroup_get_inst_coverage(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || covergroup->num_coverpoints == 0)
    return 0.0;

  // Calculate average coverage across all coverpoints for this instance
  double totalCoverage = 0.0;
  int32_t validCoverpoints = 0;

  for (int32_t i = 0; i < covergroup->num_coverpoints; ++i) {
    if (covergroup->coverpoints[i]) {
      totalCoverage += __moore_coverpoint_get_coverage(cg, i);
      validCoverpoints++;
    }
  }

  if (validCoverpoints == 0)
    return 0.0;

  return totalCoverage / validCoverpoints;
}

extern "C" double __moore_coverpoint_get_inst_coverage(void *cg,
                                                        int32_t cp_index) {
  // For coverpoints, instance coverage is the same as regular coverage
  // since coverpoints are always instance-specific
  return __moore_coverpoint_get_coverage(cg, cp_index);
}

extern "C" void __moore_coverage_report(void) {
  std::printf("\n");
  std::printf("=================================================\n");
  std::printf("          Coverage Report\n");
  std::printf("=================================================\n\n");

  if (registeredCovergroups.empty()) {
    std::printf("No covergroups registered.\n");
    std::printf("\n=================================================\n");
    return;
  }

  for (auto *cg : registeredCovergroups) {
    if (!cg)
      continue;

    double cgCoverage = __moore_covergroup_get_coverage(cg);
    std::printf("Covergroup: %s\n", cg->name ? cg->name : "(unnamed)");
    std::printf("  Overall coverage: %.2f%%\n", cgCoverage);
    std::printf("  Coverpoints: %d\n", cg->num_coverpoints);

    for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
      auto *cp = cg->coverpoints[i];
      if (!cp)
        continue;

      double cpCoverage = __moore_coverpoint_get_coverage(cg, i);
      std::printf("    - %s: %ld hits, %.2f%% coverage",
                  cp->name ? cp->name : "(unnamed)",
                  static_cast<long>(cp->hits), cpCoverage);

      // Show value range if we have samples
      if (cp->hits > 0 && cp->min_val <= cp->max_val) {
        auto trackerIt = coverpointTrackers.find(cp);
        int64_t uniqueVals = 0;
        if (trackerIt != coverpointTrackers.end()) {
          uniqueVals = static_cast<int64_t>(trackerIt->second.valueCounts.size());
        }
        std::printf(" [range: %ld..%ld, %ld unique values]",
                    static_cast<long>(cp->min_val),
                    static_cast<long>(cp->max_val),
                    static_cast<long>(uniqueVals));
      }
      std::printf("\n");
    }
    std::printf("\n");
  }

  std::printf("=================================================\n");
}

//===----------------------------------------------------------------------===//
// Explicit Bins Support
//===----------------------------------------------------------------------===//

// Note: TransitionBin, ExplicitBinData struct, and explicitBinData map are
// defined earlier in the file (around line 2136) to be accessible from
// __moore_coverpoint_get_coverage.

extern "C" void __moore_coverpoint_init_with_bins(void *cg, int32_t cp_index,
                                                   const char *name,
                                                   MooreCoverageBin *bins,
                                                   int32_t num_bins) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;

  // Allocate a new coverpoint
  auto *cp = static_cast<MooreCoverpoint *>(std::malloc(sizeof(MooreCoverpoint)));
  if (!cp)
    return;

  // Initialize the coverpoint with explicit bins
  // Make a copy of the name to avoid dangling pointers
  cp->name = name ? strdup(name) : nullptr;
  cp->num_bins = num_bins;
  cp->hits = 0;
  cp->min_val = INT64_MAX;
  cp->max_val = INT64_MIN;

  // Copy bins to internal storage
  if (num_bins > 0 && bins) {
    // Allocate hit count array (legacy format for backward compatibility)
    cp->bins = static_cast<int64_t *>(std::calloc(num_bins, sizeof(int64_t)));

    // Store explicit bin data with names and ranges
    ExplicitBinData binData;
    binData.bins.reserve(num_bins);
    for (int32_t i = 0; i < num_bins; ++i) {
      MooreCoverageBin bin = bins[i];
      bin.hit_count = 0; // Initialize hit count
      binData.bins.push_back(bin);
    }
    explicitBinData[cp] = std::move(binData);
  } else {
    cp->bins = nullptr;
  }

  // Store the coverpoint in the covergroup
  covergroup->coverpoints[cp_index] = cp;

  // Initialize the value tracker for this coverpoint
  coverpointTrackers[cp] = CoverpointTracker();
}

extern "C" void __moore_coverpoint_add_bin(void *cg, int32_t cp_index,
                                            const char *bin_name,
                                            int32_t bin_type, int64_t low,
                                            int64_t high) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return;

  // Create the new bin
  MooreCoverageBin newBin;
  newBin.name = bin_name;
  newBin.type = bin_type;
  newBin.kind = MOORE_BIN_KIND_NORMAL;  // Default to normal bin
  newBin.low = low;
  newBin.high = high;
  newBin.hit_count = 0;

  // Add to explicit bin data
  auto it = explicitBinData.find(cp);
  if (it == explicitBinData.end()) {
    ExplicitBinData binData;
    binData.bins.push_back(newBin);
    explicitBinData[cp] = std::move(binData);
  } else {
    it->second.bins.push_back(newBin);
  }

  // Update the legacy bins array
  int32_t newNumBins = cp->num_bins + 1;
  int64_t *newBins =
      static_cast<int64_t *>(std::realloc(cp->bins, newNumBins * sizeof(int64_t)));
  if (newBins) {
    newBins[cp->num_bins] = 0; // Initialize new bin hit count
    cp->bins = newBins;
    cp->num_bins = newNumBins;
  }
}

extern "C" int64_t __moore_coverpoint_get_bin_hits(void *cg, int32_t cp_index,
                                                    int32_t bin_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return 0;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp || bin_index < 0 || bin_index >= cp->num_bins)
    return 0;

  // Check explicit bin data first
  auto it = explicitBinData.find(cp);
  if (it != explicitBinData.end() && bin_index < (int32_t)it->second.bins.size()) {
    return it->second.bins[bin_index].hit_count;
  }

  // Fall back to legacy bins array
  if (cp->bins)
    return cp->bins[bin_index];

  return 0;
}

//===----------------------------------------------------------------------===//
// JSON Coverage Reporting
//===----------------------------------------------------------------------===//

namespace {

/// Helper to escape a string for JSON output.
std::string jsonEscapeString(const char *str) {
  if (!str)
    return "null";

  std::string result = "\"";
  for (const char *p = str; *p; ++p) {
    switch (*p) {
    case '"':
      result += "\\\"";
      break;
    case '\\':
      result += "\\\\";
      break;
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(*p) < 32) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x",
                      static_cast<unsigned char>(*p));
        result += buf;
      } else {
        result += *p;
      }
    }
  }
  result += "\"";
  return result;
}

/// Generate JSON coverage report as a string.
std::string generateCoverageJson() {
  std::string json = "{\n";
  json += "  \"coverage_report\": {\n";
  json += "    \"version\": \"1.0\",\n";
  json += "    \"generator\": \"circt-moore-runtime\",\n";
  json += "    \"covergroups\": [\n";

  bool firstCg = true;
  for (auto *cg : registeredCovergroups) {
    if (!cg)
      continue;

    if (!firstCg)
      json += ",\n";
    firstCg = false;

    double cgCoverage = __moore_covergroup_get_coverage(cg);

    json += "      {\n";
    json += "        \"name\": " + jsonEscapeString(cg->name) + ",\n";
    json += "        \"coverage_percent\": " + std::to_string(cgCoverage) + ",\n";
    json += "        \"num_coverpoints\": " + std::to_string(cg->num_coverpoints) + ",\n";
    json += "        \"coverpoints\": [\n";

    bool firstCp = true;
    for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
      auto *cp = cg->coverpoints[i];
      if (!cp)
        continue;

      if (!firstCp)
        json += ",\n";
      firstCp = false;

      double cpCoverage = __moore_coverpoint_get_coverage(cg, i);
      auto trackerIt = coverpointTrackers.find(cp);
      int64_t uniqueVals =
          (trackerIt != coverpointTrackers.end())
              ? static_cast<int64_t>(trackerIt->second.valueCounts.size())
              : 0;

      json += "          {\n";
      json += "            \"name\": " + jsonEscapeString(cp->name) + ",\n";
      json += "            \"coverage_percent\": " + std::to_string(cpCoverage) + ",\n";
      json += "            \"total_hits\": " + std::to_string(cp->hits) + ",\n";
      json += "            \"unique_values\": " + std::to_string(uniqueVals) + ",\n";
      json += "            \"min_value\": " + std::to_string(cp->min_val) + ",\n";
      json += "            \"max_value\": " + std::to_string(cp->max_val) + ",\n";

      // Add explicit bins if present
      json += "            \"bins\": [\n";
      auto binIt = explicitBinData.find(cp);
      if (binIt != explicitBinData.end()) {
        bool firstBin = true;
        for (const auto &bin : binIt->second.bins) {
          if (!firstBin)
            json += ",\n";
          firstBin = false;

          const char *binTypeName = "unknown";
          switch (bin.type) {
          case MOORE_BIN_VALUE:
            binTypeName = "value";
            break;
          case MOORE_BIN_RANGE:
            binTypeName = "range";
            break;
          case MOORE_BIN_WILDCARD:
            binTypeName = "wildcard";
            break;
          case MOORE_BIN_TRANSITION:
            binTypeName = "transition";
            break;
          }

          json += "              {\n";
          json += "                \"name\": " + jsonEscapeString(bin.name) + ",\n";
          json += "                \"type\": \"" + std::string(binTypeName) + "\",\n";
          json += "                \"low\": " + std::to_string(bin.low) + ",\n";
          json += "                \"high\": " + std::to_string(bin.high) + ",\n";
          json += "                \"hit_count\": " + std::to_string(bin.hit_count) + "\n";
          json += "              }";
        }
      }
      json += "\n            ],\n";

      // Add value histogram (top 10 values by frequency)
      json += "            \"top_values\": [\n";
      if (trackerIt != coverpointTrackers.end()) {
        // Sort values by count
        std::vector<std::pair<int64_t, int64_t>> sorted;
        for (const auto &kv : trackerIt->second.valueCounts) {
          sorted.emplace_back(kv.first, kv.second);
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });

        bool firstVal = true;
        int count = 0;
        for (const auto &kv : sorted) {
          if (count >= 10)
            break;
          if (!firstVal)
            json += ",\n";
          firstVal = false;
          json += "              {\"value\": " + std::to_string(kv.first) +
                  ", \"count\": " + std::to_string(kv.second) + "}";
          ++count;
        }
      }
      json += "\n            ]\n";
      json += "          }";
    }

    json += "\n        ]\n";
    json += "      }";
  }

  json += "\n    ]\n";
  json += "  }\n";
  json += "}\n";

  return json;
}

} // anonymous namespace

extern "C" int32_t __moore_coverage_report_json(const char *filename) {
  if (!filename)
    return 1;

  FILE *fp = std::fopen(filename, "w");
  if (!fp)
    return 1;

  std::string json = generateCoverageJson();
  std::fwrite(json.c_str(), 1, json.size(), fp);
  std::fclose(fp);

  return 0;
}

extern "C" void __moore_coverage_report_json_stdout(void) {
  std::string json = generateCoverageJson();
  std::printf("%s", json.c_str());
}

extern "C" char *__moore_coverage_get_json(void) {
  std::string json = generateCoverageJson();

  char *result = static_cast<char *>(std::malloc(json.size() + 1));
  if (!result)
    return nullptr;

  std::memcpy(result, json.c_str(), json.size() + 1);
  return result;
}

//===----------------------------------------------------------------------===//
// Enhanced Sampling with Explicit Bin Matching
//===----------------------------------------------------------------------===//

// Override the original sample function to also update explicit bins
// This is done by updating the coverpoint_sample function behavior
namespace {

/// Helper to check if a value matches an explicit bin.
bool matchesBin(const MooreCoverageBin &bin, int64_t value) {
  switch (bin.type) {
  case MOORE_BIN_VALUE:
    return value == bin.low;
  case MOORE_BIN_RANGE:
    return value >= bin.low && value <= bin.high;
  case MOORE_BIN_WILDCARD:
    // Wildcard matching uses mask+value encoding:
    // - bin.low = pattern value (don't care bits are 0)
    // - bin.high = mask (1 = don't care, 0 = must match)
    // Match condition: all "care" bits must match the pattern
    // Formula: (value ^ pattern) & ~mask == 0
    return ((value ^ bin.low) & ~bin.high) == 0;
  case MOORE_BIN_TRANSITION:
    // Transition matching is handled separately via transition bins
    return false;
  default:
    return false;
  }
}

/// Check if a value matches a transition step (considering wildcards and ranges).
bool valueMatchesTransitionStep(int64_t value, const MooreTransitionStep &step) {
  // For now, simple equality check. Could be extended for ranges/wildcards.
  return value == step.value;
}

/// Advance the state machine for a single sequence given a new value.
/// Returns true if the sequence completed (bin hit).
bool advanceTransitionSequenceState(TransitionBin::SequenceState &state,
                                    const MooreTransitionSequence &seq,
                                    int64_t value, int64_t prevValue,
                                    bool hasPrev) {
  if (seq.num_steps < 2)
    return false; // Need at least 2 steps for a transition

  // If not active, check if the first step matches the previous value
  // and second step matches current value (for simple transitions)
  if (!state.active) {
    // Check if we're at a potential start of the sequence
    if (hasPrev && valueMatchesTransitionStep(prevValue, seq.steps[0]) &&
        valueMatchesTransitionStep(value, seq.steps[1])) {
      if (seq.num_steps == 2) {
        // Simple 2-step transition - complete!
        return true;
      } else {
        // Start tracking longer sequence
        state.active = true;
        state.currentStep = 2;
        state.repeatCount = 0;
      }
    }
    return false;
  }

  // Active - check if current value matches the next expected step
  if (state.currentStep >= seq.num_steps) {
    state.active = false;
    return false;
  }

  const MooreTransitionStep &step = seq.steps[state.currentStep];

  // Handle repeat patterns
  if (step.repeat_kind != MOORE_TRANS_NONE) {
    if (valueMatchesTransitionStep(value, step)) {
      state.repeatCount++;

      // Check if we've seen enough repeats
      if (state.repeatCount >= step.repeat_from) {
        // Check if this could be the end of repeats (max reached or next step matches)
        if (state.repeatCount >= step.repeat_to ||
            (state.currentStep + 1 < seq.num_steps &&
             valueMatchesTransitionStep(value, seq.steps[state.currentStep + 1]))) {
          state.currentStep++;
          state.repeatCount = 0;

          // Check if sequence complete
          if (state.currentStep >= seq.num_steps) {
            state.active = false;
            return true;
          }
        }
      }
    } else {
      // Value doesn't match - check if we can move to next step
      if (state.repeatCount >= step.repeat_from &&
          state.currentStep + 1 < seq.num_steps &&
          valueMatchesTransitionStep(value, seq.steps[state.currentStep + 1])) {
        state.currentStep++;
        state.repeatCount = 0;

        if (state.currentStep >= seq.num_steps) {
          state.active = false;
          return true;
        }
      } else {
        // Sequence broken
        state.active = false;
        state.currentStep = 0;
        state.repeatCount = 0;
      }
    }
  } else {
    // No repeat - simple step matching
    if (valueMatchesTransitionStep(value, step)) {
      state.currentStep++;

      // Check if sequence complete
      if (state.currentStep >= seq.num_steps) {
        state.active = false;
        return true;
      }
    } else {
      // Sequence broken
      state.active = false;
      state.currentStep = 0;
      state.repeatCount = 0;
    }
  }

  return false;
}

/// Update transition bins when a value is sampled.
/// Returns true if any transition bin was hit.
void updateTransitionBinsHelper(MooreCoverpoint *cp, int64_t value,
                                int64_t prevValue, bool hasPrev) {
  auto it = explicitBinData.find(cp);
  if (it == explicitBinData.end())
    return;

  for (auto &transBin : it->second.transitionBins) {
    // Check each alternative sequence
    for (size_t seqIdx = 0; seqIdx < transBin.sequences.size(); ++seqIdx) {
      auto &seq = transBin.sequences[seqIdx];
      auto &seqState = transBin.sequenceStates[seqIdx];

      if (advanceTransitionSequenceState(seqState, seq, value, prevValue,
                                         hasPrev)) {
        transBin.hit_count++;
      }
    }
  }
}

/// Update explicit bins when a value is sampled.
/// This is the implementation of the forward-declared helper function.
void updateExplicitBinsHelper(MooreCoverpoint *cp, int64_t value) {
  auto it = explicitBinData.find(cp);
  if (it == explicitBinData.end())
    return;

  int32_t binIndex = 0;
  for (auto &bin : it->second.bins) {
    if (matchesBin(bin, value)) {
      bin.hit_count++;
      // Also update legacy bins array
      if (cp->bins && binIndex < cp->num_bins) {
        cp->bins[binIndex]++;
      }
    }
    ++binIndex;
  }
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Transition Coverage Operations
//===----------------------------------------------------------------------===//
//
// Transition coverage tracks state machine transitions rather than just values.
// This implements SystemVerilog transition bins like: bins x = (IDLE => RUN);
//

namespace {

/// Structure to store a transition bin definition.
struct TransitionBinDef {
  const char *name;
  std::vector<MooreTransitionSequence> sequences; // Alternative sequences
  int64_t hit_count;
};

/// Structure to track the matching state for a single transition sequence.
struct TransitionSequenceState {
  int32_t current_step;  // Which step in the sequence we're waiting for
  int32_t repeat_count;  // For repeat patterns, count of matching values
  bool active;           // Is this sequence currently being tracked
};

/// Structure to store transition tracking state for a coverpoint.
struct TransitionTrackerState {
  MooreCovergroup *covergroup;
  int32_t cp_index;
  std::vector<TransitionBinDef> bins;

  // For each bin, for each alternative sequence, track the matching state
  // Indexed as: sequenceStates[bin_index][sequence_index]
  std::vector<std::vector<TransitionSequenceState>> sequenceStates;

  // Previous value for detecting transitions
  int64_t prev_value;
  bool has_prev_value;
};

/// Check if a value matches a transition step (considering wildcards and ranges).
bool valueMatchesStep(int64_t value, const MooreTransitionStep &step) {
  // For now, simple equality check. Could be extended for ranges/wildcards.
  return value == step.value;
}

/// Advance the state machine for a single sequence given a new value.
/// Returns true if the sequence completed (bin hit).
bool advanceSequenceState(TransitionSequenceState &state,
                          const MooreTransitionSequence &seq,
                          int64_t value, int64_t prev_value, bool has_prev) {
  if (seq.num_steps < 2)
    return false; // Need at least 2 steps for a transition

  // If not active, check if the first step matches the previous value
  // and second step matches current value (for simple transitions)
  if (!state.active) {
    // Check if we're at a potential start of the sequence
    if (has_prev && valueMatchesStep(prev_value, seq.steps[0]) &&
        valueMatchesStep(value, seq.steps[1])) {
      if (seq.num_steps == 2) {
        // Simple 2-step transition - complete!
        return true;
      } else {
        // Start tracking longer sequence
        state.active = true;
        state.current_step = 2;
        state.repeat_count = 0;
      }
    }
    return false;
  }

  // Active - check if current value matches the next expected step
  if (state.current_step >= seq.num_steps) {
    state.active = false;
    return false;
  }

  const MooreTransitionStep &step = seq.steps[state.current_step];

  // Handle repeat patterns
  if (step.repeat_kind != MOORE_TRANS_NONE) {
    if (valueMatchesStep(value, step)) {
      state.repeat_count++;

      // Check if we've seen enough repeats
      if (state.repeat_count >= step.repeat_from) {
        // Check if this could be the end of repeats (max reached or next step matches)
        if (state.repeat_count >= step.repeat_to ||
            (state.current_step + 1 < seq.num_steps &&
             valueMatchesStep(value, seq.steps[state.current_step + 1]))) {
          state.current_step++;
          state.repeat_count = 0;

          // Check if sequence complete
          if (state.current_step >= seq.num_steps) {
            state.active = false;
            return true;
          }
        }
      }
    } else {
      // Value doesn't match - check if we can move to next step
      if (state.repeat_count >= step.repeat_from &&
          state.current_step + 1 < seq.num_steps &&
          valueMatchesStep(value, seq.steps[state.current_step + 1])) {
        state.current_step++;
        state.repeat_count = 0;

        if (state.current_step >= seq.num_steps) {
          state.active = false;
          return true;
        }
      } else {
        // Sequence broken
        state.active = false;
        state.current_step = 0;
        state.repeat_count = 0;
      }
    }
  } else {
    // No repeat - simple step matching
    if (valueMatchesStep(value, step)) {
      state.current_step++;

      // Check if sequence complete
      if (state.current_step >= seq.num_steps) {
        state.active = false;
        return true;
      }
    } else {
      // Sequence broken
      state.active = false;
      state.current_step = 0;
      state.repeat_count = 0;
    }
  }

  return false;
}

} // anonymous namespace

/// The actual transition tracker structure.
struct MooreTransitionTracker {
  TransitionTrackerState state;
};

extern "C" MooreTransitionTrackerHandle
__moore_transition_tracker_create(void *cg, int32_t cp_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return nullptr;

  auto *tracker = new MooreTransitionTracker();
  tracker->state.covergroup = covergroup;
  tracker->state.cp_index = cp_index;
  tracker->state.prev_value = 0;
  tracker->state.has_prev_value = false;

  return tracker;
}

extern "C" void
__moore_transition_tracker_destroy(MooreTransitionTrackerHandle tracker) {
  if (tracker) {
    // Free the sequence steps
    for (auto &bin : tracker->state.bins) {
      for (auto &seq : bin.sequences) {
        if (seq.steps) {
          std::free(seq.steps);
        }
      }
    }
    delete tracker;
  }
}

extern "C" void __moore_coverpoint_add_transition_bin(
    void *cg, int32_t cp_index, const char *bin_name,
    MooreTransitionSequence *sequences, int32_t num_sequences) {
  // This function is called to define transition bins on a coverpoint.
  // Store the bin definition in explicitBinData for integrated tracking.
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return;

  if (!sequences || num_sequences <= 0)
    return;

  // Create a new transition bin
  TransitionBin transBin;
  transBin.name = bin_name;
  transBin.hit_count = 0;

  // Copy the sequences
  for (int32_t i = 0; i < num_sequences; ++i) {
    MooreTransitionSequence seq;
    seq.num_steps = sequences[i].num_steps;
    // Deep copy the steps array
    if (seq.num_steps > 0 && sequences[i].steps) {
      seq.steps = static_cast<MooreTransitionStep *>(
          std::malloc(seq.num_steps * sizeof(MooreTransitionStep)));
      if (seq.steps) {
        std::memcpy(seq.steps, sequences[i].steps,
                    seq.num_steps * sizeof(MooreTransitionStep));
      }
    } else {
      seq.steps = nullptr;
    }
    transBin.sequences.push_back(seq);
    // Initialize sequence state for this alternative
    transBin.sequenceStates.push_back(TransitionBin::SequenceState());
  }

  // Add to explicit bin data
  auto it = explicitBinData.find(cp);
  if (it == explicitBinData.end()) {
    ExplicitBinData binData;
    binData.transitionBins.push_back(std::move(transBin));
    explicitBinData[cp] = std::move(binData);
  } else {
    it->second.transitionBins.push_back(std::move(transBin));
  }
}

extern "C" void
__moore_transition_tracker_sample(MooreTransitionTrackerHandle tracker,
                                  int64_t value) {
  if (!tracker)
    return;

  auto &state = tracker->state;

  // Check each bin's sequences for matches
  for (size_t binIdx = 0; binIdx < state.bins.size(); ++binIdx) {
    auto &bin = state.bins[binIdx];
    auto &binStates = state.sequenceStates[binIdx];

    for (size_t seqIdx = 0; seqIdx < bin.sequences.size(); ++seqIdx) {
      auto &seq = bin.sequences[seqIdx];
      auto &seqState = binStates[seqIdx];

      if (advanceSequenceState(seqState, seq, value, state.prev_value,
                               state.has_prev_value)) {
        bin.hit_count++;
      }
    }
  }

  // Update previous value
  state.prev_value = value;
  state.has_prev_value = true;
}

extern "C" void
__moore_transition_tracker_reset(MooreTransitionTrackerHandle tracker) {
  if (!tracker)
    return;

  auto &state = tracker->state;
  state.prev_value = 0;
  state.has_prev_value = false;

  // Reset all sequence states
  for (auto &binStates : state.sequenceStates) {
    for (auto &seqState : binStates) {
      seqState.active = false;
      seqState.current_step = 0;
      seqState.repeat_count = 0;
    }
  }
}

extern "C" int64_t __moore_transition_bin_get_hits(void *cg, int32_t cp_index,
                                                   int32_t bin_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return 0;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return 0;

  // Look up transition bins in explicitBinData
  auto it = explicitBinData.find(cp);
  if (it == explicitBinData.end())
    return 0;

  if (bin_index < 0 ||
      bin_index >= static_cast<int32_t>(it->second.transitionBins.size()))
    return 0;

  return it->second.transitionBins[bin_index].hit_count;
}

//===----------------------------------------------------------------------===//
// Cross Coverage Operations
//===----------------------------------------------------------------------===//
//
// Cross coverage tracks combinations of values from multiple coverpoints.
// This implements the SystemVerilog `cross` construct within covergroups.
//

namespace {

/// Internal representation of a named cross bin with filters.
struct CrossNamedBin {
  std::string name;
  int32_t kind; // MooreCrossBinKind
  std::vector<MooreCrossBinsofFilter> filters;
  int64_t hit_count = 0;
};

/// Extended cross coverage data per cross item.
struct CrossItemData {
  std::vector<CrossNamedBin> namedBins;
  /// Map from per-cross bin index to named bins that match it.
  /// Used for fast lookup during sampling.
};

/// Structure to store cross coverage data for a covergroup.
struct CrossCoverageData {
  std::vector<MooreCrossCoverage> crosses;
  /// Map from cross index to a map of value tuples to hit counts.
  /// The key is a vector of int64_t values (one per coverpoint in the cross).
  std::map<std::vector<int64_t>, int64_t> crossBins;
  /// Named bins for each cross (indexed by cross index).
  std::vector<CrossItemData> crossItemData;
};

/// Map from covergroup to its cross coverage data.
thread_local std::map<MooreCovergroup *, CrossCoverageData> crossCoverageData;

/// Global illegal cross bin callback.
thread_local MooreIllegalCrossBinCallback illegalCrossBinCallback = nullptr;
thread_local void *illegalCrossBinCallbackUserData = nullptr;

/// Map from covergroup to its coverage goal (default 100.0).
thread_local std::map<MooreCovergroup *, double> covergroupGoals;

// Note: CovergroupOptions and CoverpointOptions structs and their maps are
// defined earlier in the file (around line 2112) to be accessible from
// __moore_coverpoint_get_coverage.

/// Helper function to check if a value matches a binsof filter.
/// Returns true if the value satisfies the filter condition.
bool matchesBinsofFilter(const MooreCrossBinsofFilter &filter, int64_t value,
                         MooreCovergroup *covergroup) {
  bool matches = false;

  // If no specific bins or values are specified, the filter matches all values
  if (filter.num_bins == 0 && filter.num_values == 0) {
    matches = true;
  } else {
    // Check if value matches any of the specified bin indices
    if (filter.num_bins > 0 && filter.bin_indices) {
      // Get the coverpoint's explicit bins
      int32_t cpIdx = filter.cp_index;
      if (cpIdx >= 0 && cpIdx < covergroup->num_coverpoints) {
        auto *cp = covergroup->coverpoints[cpIdx];
        if (cp) {
          // Check if value falls within any of the specified bin ranges
          auto binIt = explicitBinData.find(cp);
          if (binIt != explicitBinData.end()) {
            for (int32_t i = 0; i < filter.num_bins; ++i) {
              int32_t binIdx = filter.bin_indices[i];
              if (binIdx >= 0 &&
                  binIdx < static_cast<int32_t>(binIt->second.bins.size())) {
                const auto &bin = binIt->second.bins[binIdx];
                if (bin.type == MOORE_BIN_VALUE) {
                  if (value == bin.low) {
                    matches = true;
                    break;
                  }
                } else if (bin.type == MOORE_BIN_RANGE) {
                  if (value >= bin.low && value <= bin.high) {
                    matches = true;
                    break;
                  }
                }
              }
            }
          }
        }
      }
    }

    // Check if value matches any of the intersect values
    if (filter.num_values > 0 && filter.values) {
      for (int32_t i = 0; i < filter.num_values; ++i) {
        if (value == filter.values[i]) {
          matches = true;
          break;
        }
      }
    }
  }

  // Apply negation if needed
  return filter.negate ? !matches : matches;
}

/// Helper function to check if a value tuple matches a named cross bin.
bool matchesNamedCrossBin(const CrossNamedBin &namedBin,
                          const std::vector<int64_t> &values,
                          MooreCovergroup *covergroup,
                          const MooreCrossCoverage &cross) {
  // All filters must match (AND semantics)
  for (const auto &filter : namedBin.filters) {
    // Find the index of this coverpoint in the cross
    int32_t crossCpIdx = -1;
    for (int32_t i = 0; i < cross.num_cps; ++i) {
      if (cross.cp_indices[i] == filter.cp_index) {
        crossCpIdx = i;
        break;
      }
    }

    if (crossCpIdx < 0 ||
        crossCpIdx >= static_cast<int32_t>(values.size())) {
      // Filter references a coverpoint not in this cross
      return false;
    }

    int64_t value = values[crossCpIdx];
    if (!matchesBinsofFilter(filter, value, covergroup)) {
      return false;
    }
  }

  return true;
}

} // anonymous namespace

extern "C" int32_t __moore_cross_create(void *cg, const char *name,
                                        int32_t *cp_indices, int32_t num_cps) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || !cp_indices || num_cps < 2)
    return -1;

  // Validate coverpoint indices
  for (int32_t i = 0; i < num_cps; ++i) {
    if (cp_indices[i] < 0 || cp_indices[i] >= covergroup->num_coverpoints)
      return -1;
  }

  // Get or create cross coverage data for this covergroup
  auto &crossData = crossCoverageData[covergroup];

  // Create the cross
  MooreCrossCoverage cross;
  cross.name = name;
  cross.num_cps = num_cps;

  // Copy the coverpoint indices
  cross.cp_indices = static_cast<int32_t *>(std::malloc(num_cps * sizeof(int32_t)));
  if (!cross.cp_indices)
    return -1;
  std::memcpy(cross.cp_indices, cp_indices, num_cps * sizeof(int32_t));

  // bins_data will be used to store the cross bin map
  cross.bins_data = nullptr;

  int32_t crossIndex = static_cast<int32_t>(crossData.crosses.size());
  crossData.crosses.push_back(cross);

  // Also create the CrossItemData for this cross
  crossData.crossItemData.push_back(CrossItemData());

  return crossIndex;
}

extern "C" void __moore_cross_sample(void *cg, int64_t *cp_values,
                                     int32_t num_values) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || !cp_values)
    return;

  auto it = crossCoverageData.find(covergroup);
  if (it == crossCoverageData.end())
    return;

  // For each cross, record the combination of values
  for (size_t crossIdx = 0; crossIdx < it->second.crosses.size(); ++crossIdx) {
    auto &cross = it->second.crosses[crossIdx];
    std::vector<int64_t> valueKey;
    valueKey.reserve(cross.num_cps);

    bool validSample = true;
    for (int32_t i = 0; i < cross.num_cps; ++i) {
      int32_t cpIdx = cross.cp_indices[i];
      if (cpIdx < 0 || cpIdx >= num_values) {
        validSample = false;
        break;
      }
      valueKey.push_back(cp_values[cpIdx]);
    }

    if (validSample) {
      // Check if this sample matches any named cross bins
      if (crossIdx < it->second.crossItemData.size()) {
        auto &itemData = it->second.crossItemData[crossIdx];

        // Check for illegal bins first
        bool isIllegal = false;
        const char *illegalBinName = nullptr;
        for (auto &namedBin : itemData.namedBins) {
          if (namedBin.kind == MOORE_CROSS_BIN_ILLEGAL) {
            if (matchesNamedCrossBin(namedBin, valueKey, covergroup, cross)) {
              isIllegal = true;
              illegalBinName = namedBin.name.c_str();
              namedBin.hit_count++;

              // Invoke the callback if registered
              if (illegalCrossBinCallback) {
                illegalCrossBinCallback(covergroup->name, cross.name,
                                        illegalBinName, valueKey.data(),
                                        cross.num_cps,
                                        illegalCrossBinCallbackUserData);
              }
              break;
            }
          }
        }

        // Check for ignore bins - skip sampling if matched
        bool isIgnored = false;
        for (const auto &namedBin : itemData.namedBins) {
          if (namedBin.kind == MOORE_CROSS_BIN_IGNORE) {
            if (matchesNamedCrossBin(namedBin, valueKey, covergroup, cross)) {
              isIgnored = true;
              break;
            }
          }
        }

        // Update hit counts for normal named bins
        if (!isIllegal && !isIgnored) {
          for (auto &namedBin : itemData.namedBins) {
            if (namedBin.kind == MOORE_CROSS_BIN_NORMAL) {
              if (matchesNamedCrossBin(namedBin, valueKey, covergroup, cross)) {
                namedBin.hit_count++;
              }
            }
          }
        }

        // Only record the cross bin if not ignored
        if (!isIgnored) {
          it->second.crossBins[valueKey]++;
        }
      } else {
        // No named bins defined, just record the sample
        it->second.crossBins[valueKey]++;
      }
    }
  }
}

extern "C" double __moore_cross_get_coverage(void *cg, int32_t cross_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return 0.0;

  auto it = crossCoverageData.find(covergroup);
  if (it == crossCoverageData.end() || cross_index < 0 ||
      cross_index >= static_cast<int32_t>(it->second.crosses.size()))
    return 0.0;

  // Get the at_least threshold from the covergroup options
  int64_t atLeast = 1;
  auto cgOptIt = covergroupOptions.find(covergroup);
  if (cgOptIt != covergroupOptions.end()) {
    atLeast = cgOptIt->second.atLeast;
  }

  // Count unique cross bins hit for this cross
  const auto &cross = it->second.crosses[cross_index];
  int64_t binsHit = 0;
  int64_t totalPossibleBins = 1;

  // Calculate total possible cross bins from coverpoint ranges
  // For each coverpoint in the cross, multiply by its unique value count
  for (int32_t i = 0; i < cross.num_cps; ++i) {
    int32_t cpIdx = cross.cp_indices[i];
    if (cpIdx >= 0 && cpIdx < covergroup->num_coverpoints) {
      auto *cp = covergroup->coverpoints[cpIdx];
      if (cp) {
        auto trackerIt = coverpointTrackers.find(cp);
        if (trackerIt != coverpointTrackers.end()) {
          int64_t uniqueVals = static_cast<int64_t>(
              trackerIt->second.valueCounts.size());
          if (uniqueVals > 0)
            totalPossibleBins *= uniqueVals;
        }
      }
    }
  }

  // Count actual cross bins hit for this specific cross
  // A bin is considered covered only if its hit count >= at_least
  for (const auto &kv : it->second.crossBins) {
    // Check if this bin belongs to this cross by comparing size
    if (kv.first.size() == static_cast<size_t>(cross.num_cps)) {
      // Check if this bin meets the at_least threshold
      if (kv.second >= atLeast) {
        binsHit++;
      }
    }
  }

  if (totalPossibleBins == 0)
    return 0.0;

  double coverage = (100.0 * binsHit) / totalPossibleBins;
  return coverage > 100.0 ? 100.0 : coverage;
}

extern "C" double __moore_cross_get_inst_coverage(void *cg, int32_t cross_index) {
  // For crosses, instance coverage is the same as regular coverage
  // since crosses are always instance-specific within their covergroup
  return __moore_cross_get_coverage(cg, cross_index);
}

extern "C" int64_t __moore_cross_get_bins_hit(void *cg, int32_t cross_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return 0;

  auto it = crossCoverageData.find(covergroup);
  if (it == crossCoverageData.end() || cross_index < 0 ||
      cross_index >= static_cast<int32_t>(it->second.crosses.size()))
    return 0;

  const auto &cross = it->second.crosses[cross_index];
  int64_t binsHit = 0;

  for (const auto &kv : it->second.crossBins) {
    if (kv.first.size() == static_cast<size_t>(cross.num_cps)) {
      binsHit++;
    }
  }

  return binsHit;
}

//===----------------------------------------------------------------------===//
// Cross Coverage Named Bins and Filtering
//===----------------------------------------------------------------------===//

extern "C" int32_t __moore_cross_add_named_bin(void *cg, int32_t cross_index,
                                                const char *name, int32_t kind,
                                                MooreCrossBinsofFilter *filters,
                                                int32_t num_filters) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cross_index < 0 || !name)
    return -1;

  auto it = crossCoverageData.find(covergroup);
  if (it == crossCoverageData.end() ||
      cross_index >= static_cast<int32_t>(it->second.crosses.size()))
    return -1;

  // Create the named bin
  CrossNamedBin namedBin;
  namedBin.name = name;
  namedBin.kind = kind;
  namedBin.hit_count = 0;

  // Copy filters
  if (filters && num_filters > 0) {
    namedBin.filters.reserve(num_filters);
    for (int32_t i = 0; i < num_filters; ++i) {
      namedBin.filters.push_back(filters[i]);
    }
  }

  // Ensure we have enough crossItemData entries
  while (it->second.crossItemData.size() <=
         static_cast<size_t>(cross_index)) {
    it->second.crossItemData.push_back(CrossItemData());
  }

  auto &itemData = it->second.crossItemData[cross_index];
  int32_t binIndex = static_cast<int32_t>(itemData.namedBins.size());
  itemData.namedBins.push_back(std::move(namedBin));

  return binIndex;
}

extern "C" int32_t __moore_cross_add_ignore_bin(void *cg, int32_t cross_index,
                                                 const char *name,
                                                 MooreCrossBinsofFilter *filters,
                                                 int32_t num_filters) {
  return __moore_cross_add_named_bin(cg, cross_index, name,
                                      MOORE_CROSS_BIN_IGNORE, filters,
                                      num_filters);
}

extern "C" int32_t __moore_cross_add_illegal_bin(void *cg, int32_t cross_index,
                                                  const char *name,
                                                  MooreCrossBinsofFilter *filters,
                                                  int32_t num_filters) {
  return __moore_cross_add_named_bin(cg, cross_index, name,
                                      MOORE_CROSS_BIN_ILLEGAL, filters,
                                      num_filters);
}

extern "C" int64_t __moore_cross_get_named_bin_hits(void *cg,
                                                     int32_t cross_index,
                                                     int32_t bin_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cross_index < 0 || bin_index < 0)
    return 0;

  auto it = crossCoverageData.find(covergroup);
  if (it == crossCoverageData.end() ||
      cross_index >= static_cast<int32_t>(it->second.crossItemData.size()))
    return 0;

  const auto &itemData = it->second.crossItemData[cross_index];
  if (bin_index >= static_cast<int32_t>(itemData.namedBins.size()))
    return 0;

  return itemData.namedBins[bin_index].hit_count;
}

extern "C" bool __moore_cross_is_illegal(void *cg, int32_t cross_index,
                                          int64_t *values) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cross_index < 0 || !values)
    return false;

  auto it = crossCoverageData.find(covergroup);
  if (it == crossCoverageData.end() ||
      cross_index >= static_cast<int32_t>(it->second.crosses.size()) ||
      cross_index >= static_cast<int32_t>(it->second.crossItemData.size()))
    return false;

  const auto &cross = it->second.crosses[cross_index];
  const auto &itemData = it->second.crossItemData[cross_index];

  // Build value vector
  std::vector<int64_t> valueVec(values, values + cross.num_cps);

  // Check if any illegal bin matches
  for (const auto &namedBin : itemData.namedBins) {
    if (namedBin.kind == MOORE_CROSS_BIN_ILLEGAL) {
      if (matchesNamedCrossBin(namedBin, valueVec, covergroup, cross)) {
        return true;
      }
    }
  }

  return false;
}

extern "C" bool __moore_cross_is_ignored(void *cg, int32_t cross_index,
                                          int64_t *values) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cross_index < 0 || !values)
    return false;

  auto it = crossCoverageData.find(covergroup);
  if (it == crossCoverageData.end() ||
      cross_index >= static_cast<int32_t>(it->second.crosses.size()) ||
      cross_index >= static_cast<int32_t>(it->second.crossItemData.size()))
    return false;

  const auto &cross = it->second.crosses[cross_index];
  const auto &itemData = it->second.crossItemData[cross_index];

  // Build value vector
  std::vector<int64_t> valueVec(values, values + cross.num_cps);

  // Check if any ignore bin matches
  for (const auto &namedBin : itemData.namedBins) {
    if (namedBin.kind == MOORE_CROSS_BIN_IGNORE) {
      if (matchesNamedCrossBin(namedBin, valueVec, covergroup, cross)) {
        return true;
      }
    }
  }

  return false;
}

extern "C" int32_t __moore_cross_get_num_named_bins(void *cg,
                                                     int32_t cross_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cross_index < 0)
    return 0;

  auto it = crossCoverageData.find(covergroup);
  if (it == crossCoverageData.end() ||
      cross_index >= static_cast<int32_t>(it->second.crossItemData.size()))
    return 0;

  return static_cast<int32_t>(it->second.crossItemData[cross_index].namedBins.size());
}

extern "C" void __moore_cross_set_illegal_bin_callback(
    MooreIllegalCrossBinCallback callback, void *userData) {
  illegalCrossBinCallback = callback;
  illegalCrossBinCallbackUserData = userData;
}

/// Internal function to clean up cross coverage data for a covergroup.
/// This is called from __moore_covergroup_destroy.
void __moore_cross_cleanup_for_covergroup(MooreCovergroup *covergroup) {
  auto it = crossCoverageData.find(covergroup);
  if (it != crossCoverageData.end()) {
    // Free cp_indices for each cross
    for (auto &cross : it->second.crosses) {
      if (cross.cp_indices) {
        std::free(cross.cp_indices);
      }
    }
    crossCoverageData.erase(it);
  }
}

//===----------------------------------------------------------------------===//
// Coverage Reset and Aggregation
//===----------------------------------------------------------------------===//

extern "C" void __moore_covergroup_reset(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;

  // Reset each coverpoint
  for (int32_t i = 0; i < covergroup->num_coverpoints; ++i) {
    __moore_coverpoint_reset(cg, i);
  }

  // Reset cross coverage data
  auto crossIt = crossCoverageData.find(covergroup);
  if (crossIt != crossCoverageData.end()) {
    crossIt->second.crossBins.clear();
    // Reset named cross bin hit counts
    for (auto &itemData : crossIt->second.crossItemData) {
      for (auto &namedBin : itemData.namedBins) {
        namedBin.hit_count = 0;
      }
    }
  }
}

extern "C" void __moore_coverpoint_reset(void *cg, int32_t cp_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return;

  // Reset hit count and range tracking
  cp->hits = 0;
  cp->min_val = INT64_MAX;
  cp->max_val = INT64_MIN;

  // Reset explicit bin hit counts
  if (cp->bins) {
    for (int32_t i = 0; i < cp->num_bins; ++i) {
      cp->bins[i] = 0;
    }
  }

  // Reset explicit bin data
  auto binIt = explicitBinData.find(cp);
  if (binIt != explicitBinData.end()) {
    for (auto &bin : binIt->second.bins) {
      bin.hit_count = 0;
    }
    // Reset transition bins
    for (auto &transBin : binIt->second.transitionBins) {
      transBin.hit_count = 0;
      // Reset sequence states
      for (auto &seqState : transBin.sequenceStates) {
        seqState.active = false;
        seqState.currentStep = 0;
        seqState.repeatCount = 0;
      }
    }
  }

  // Reset value tracker (including previous value for transitions)
  auto trackerIt = coverpointTrackers.find(cp);
  if (trackerIt != coverpointTrackers.end()) {
    trackerIt->second.valueCounts.clear();
    trackerIt->second.prevValue = 0;
    trackerIt->second.hasPrevValue = false;
  }
}

extern "C" double __moore_coverage_get_total(void) {
  if (registeredCovergroups.empty())
    return 0.0;

  double totalCoverage = 0.0;
  int32_t validGroups = 0;

  for (auto *cg : registeredCovergroups) {
    if (cg) {
      totalCoverage += __moore_covergroup_get_coverage(cg);
      validGroups++;
    }
  }

  if (validGroups == 0)
    return 0.0;

  return totalCoverage / validGroups;
}

extern "C" int32_t __moore_coverage_get_num_covergroups(void) {
  return static_cast<int32_t>(registeredCovergroups.size());
}

extern "C" void __moore_covergroup_set_goal(void *cg, double goal) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;

  // Clamp goal to valid range
  if (goal < 0.0)
    goal = 0.0;
  if (goal > 100.0)
    goal = 100.0;

  covergroupGoals[covergroup] = goal;
}

extern "C" double __moore_covergroup_get_goal(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return 100.0;

  auto it = covergroupGoals.find(covergroup);
  if (it != covergroupGoals.end())
    return it->second;

  return 100.0; // Default goal
}

extern "C" bool __moore_covergroup_goal_met(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return false;

  double coverage = __moore_covergroup_get_coverage(cg);
  double goal = __moore_covergroup_get_goal(cg);

  return coverage >= goal;
}

//===----------------------------------------------------------------------===//
// Coverage Assertion APIs
//===----------------------------------------------------------------------===//
//
// These functions provide assertion-style coverage checking that can be used
// to enforce coverage goals during simulation. When assertions fail, they
// can invoke a user-registered callback.
//

namespace {

/// State for coverage assertion functionality.
struct CoverageAssertionState {
  /// Callback for assertion failures.
  MooreCoverageAssertCallback failureCallback = nullptr;
  void *failureCallbackUserData = nullptr;

  /// Registered assertions for end-of-simulation checking.
  struct RegisteredAssertion {
    MooreCovergroup *covergroup; // NULL for global coverage check
    int32_t coverpointIndex;     // -1 for covergroup-level check
    double minPercentage;
  };
  std::vector<RegisteredAssertion> registeredAssertions;
};

thread_local CoverageAssertionState coverageAssertionState;

/// Helper to invoke the failure callback if set.
void invokeFailureCallback(const char *cgName, const char *cpName,
                           double actualCoverage, double requiredGoal) {
  if (coverageAssertionState.failureCallback) {
    coverageAssertionState.failureCallback(
        cgName, cpName, actualCoverage, requiredGoal,
        coverageAssertionState.failureCallbackUserData);
  }
}

} // namespace

extern "C" void __moore_coverage_set_failure_callback(
    MooreCoverageAssertCallback callback, void *userData) {
  coverageAssertionState.failureCallback = callback;
  coverageAssertionState.failureCallbackUserData = userData;
}

extern "C" bool __moore_coverage_assert_goal(double min_percentage) {
  // Clamp percentage to valid range
  if (min_percentage < 0.0)
    min_percentage = 0.0;
  if (min_percentage > 100.0)
    min_percentage = 100.0;

  double totalCoverage = __moore_coverage_get_total();

  if (totalCoverage >= min_percentage) {
    return true;
  }

  // Assertion failed - invoke callback
  invokeFailureCallback(nullptr, nullptr, totalCoverage, min_percentage);
  return false;
}

extern "C" bool __moore_covergroup_assert_goal(void *cg, double min_percentage) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return false;

  // Clamp percentage to valid range
  if (min_percentage < 0.0)
    min_percentage = 0.0;
  if (min_percentage > 100.0)
    min_percentage = 100.0;

  // Use the higher of the specified percentage and the covergroup's configured goal
  double configuredGoal = __moore_covergroup_get_goal(cg);
  double effectiveGoal = std::max(min_percentage, configuredGoal);

  double coverage = __moore_covergroup_get_coverage(cg);

  if (coverage >= effectiveGoal) {
    return true;
  }

  // Assertion failed - invoke callback
  invokeFailureCallback(covergroup->name, nullptr, coverage, effectiveGoal);
  return false;
}

extern "C" bool __moore_coverpoint_assert_goal(void *cg, int32_t cp_index,
                                               double min_percentage) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return false;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return false;

  // Clamp percentage to valid range
  if (min_percentage < 0.0)
    min_percentage = 0.0;
  if (min_percentage > 100.0)
    min_percentage = 100.0;

  // Use the higher of the specified percentage and the coverpoint's configured goal
  double configuredGoal = __moore_coverpoint_get_goal(cg, cp_index);
  double effectiveGoal = std::max(min_percentage, configuredGoal);

  double coverage = __moore_coverpoint_get_coverage(cg, cp_index);

  if (coverage >= effectiveGoal) {
    return true;
  }

  // Assertion failed - invoke callback
  invokeFailureCallback(covergroup->name, cp->name, coverage, effectiveGoal);
  return false;
}

extern "C" bool __moore_coverage_check_all_goals(void) {
  bool allGoalsMet = true;

  // Check all registered covergroups
  for (auto *cg : registeredCovergroups) {
    if (!cg)
      continue;

    // Check covergroup-level goal
    double cgCoverage = __moore_covergroup_get_coverage(cg);
    double cgGoal = __moore_covergroup_get_goal(cg);

    if (cgCoverage < cgGoal) {
      invokeFailureCallback(cg->name, nullptr, cgCoverage, cgGoal);
      allGoalsMet = false;
    }

    // Check each coverpoint's goal
    for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
      auto *cp = cg->coverpoints[i];
      if (!cp)
        continue;

      double cpCoverage = __moore_coverpoint_get_coverage(cg, i);
      double cpGoal = __moore_coverpoint_get_goal(cg, i);

      if (cpCoverage < cpGoal) {
        invokeFailureCallback(cg->name, cp->name, cpCoverage, cpGoal);
        allGoalsMet = false;
      }
    }
  }

  return allGoalsMet;
}

extern "C" int32_t __moore_coverage_get_unmet_goal_count(void) {
  int32_t unmetCount = 0;

  // Count unmet goals in all registered covergroups
  for (auto *cg : registeredCovergroups) {
    if (!cg)
      continue;

    // Check covergroup-level goal
    double cgCoverage = __moore_covergroup_get_coverage(cg);
    double cgGoal = __moore_covergroup_get_goal(cg);

    if (cgCoverage < cgGoal) {
      unmetCount++;
    }

    // Check each coverpoint's goal
    for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
      auto *cp = cg->coverpoints[i];
      if (!cp)
        continue;

      double cpCoverage = __moore_coverpoint_get_coverage(cg, i);
      double cpGoal = __moore_coverpoint_get_goal(cg, i);

      if (cpCoverage < cpGoal) {
        unmetCount++;
      }
    }
  }

  return unmetCount;
}

extern "C" int32_t __moore_coverage_register_assertion(void *cg, int32_t cp_index,
                                                       double min_percentage) {
  // Clamp percentage to valid range
  if (min_percentage < 0.0)
    min_percentage = 0.0;
  if (min_percentage > 100.0)
    min_percentage = 100.0;

  // Validate covergroup if specified
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (cg && !covergroup)
    return -1;

  // Validate coverpoint index if specified
  if (covergroup && cp_index >= 0) {
    if (cp_index >= covergroup->num_coverpoints)
      return -1;
    if (!covergroup->coverpoints[cp_index])
      return -1;
  }

  // Register the assertion
  CoverageAssertionState::RegisteredAssertion assertion;
  assertion.covergroup = covergroup;
  assertion.coverpointIndex = cp_index;
  assertion.minPercentage = min_percentage;

  coverageAssertionState.registeredAssertions.push_back(assertion);

  // Return the assertion ID (index in the vector)
  return static_cast<int32_t>(
      coverageAssertionState.registeredAssertions.size() - 1);
}

extern "C" bool __moore_coverage_check_registered_assertions(void) {
  bool allPassed = true;

  for (const auto &assertion : coverageAssertionState.registeredAssertions) {
    bool passed = false;

    if (!assertion.covergroup) {
      // Global coverage check
      passed = __moore_coverage_assert_goal(assertion.minPercentage);
    } else if (assertion.coverpointIndex < 0) {
      // Covergroup-level check
      passed = __moore_covergroup_assert_goal(assertion.covergroup,
                                              assertion.minPercentage);
    } else {
      // Coverpoint-level check
      passed = __moore_coverpoint_assert_goal(
          assertion.covergroup, assertion.coverpointIndex,
          assertion.minPercentage);
    }

    if (!passed) {
      allPassed = false;
    }
  }

  return allPassed;
}

extern "C" void __moore_coverage_clear_registered_assertions(void) {
  coverageAssertionState.registeredAssertions.clear();
}

//===----------------------------------------------------------------------===//
// Coverage Options - Covergroup Level
//===----------------------------------------------------------------------===//

extern "C" void __moore_covergroup_set_weight(void *cg, int64_t weight) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;
  covergroupOptions[covergroup].weight = weight > 0 ? weight : 1;
}

extern "C" int64_t __moore_covergroup_get_weight(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return 1;
  auto it = covergroupOptions.find(covergroup);
  return it != covergroupOptions.end() ? it->second.weight : 1;
}

extern "C" void __moore_covergroup_set_per_instance(void *cg, bool perInstance) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;
  covergroupOptions[covergroup].perInstance = perInstance;
}

extern "C" bool __moore_covergroup_get_per_instance(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return false;
  auto it = covergroupOptions.find(covergroup);
  return it != covergroupOptions.end() ? it->second.perInstance : false;
}

extern "C" void __moore_covergroup_set_at_least(void *cg, int64_t atLeast) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;
  covergroupOptions[covergroup].atLeast = atLeast > 0 ? atLeast : 1;
}

extern "C" int64_t __moore_covergroup_get_at_least(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return 1;
  auto it = covergroupOptions.find(covergroup);
  return it != covergroupOptions.end() ? it->second.atLeast : 1;
}

extern "C" void __moore_covergroup_set_comment(void *cg, const char *comment) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;
  covergroupOptions[covergroup].comment = comment ? comment : "";
}

extern "C" const char *__moore_covergroup_get_comment(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return nullptr;
  auto it = covergroupOptions.find(covergroup);
  if (it != covergroupOptions.end() && !it->second.comment.empty())
    return it->second.comment.c_str();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Coverage Options - Coverpoint Level
//===----------------------------------------------------------------------===//

extern "C" void __moore_coverpoint_set_weight(void *cg, int32_t cp_index,
                                               int64_t weight) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;
  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return;
  coverpointOptions[cp].weight = weight > 0 ? weight : 1;
}

extern "C" int64_t __moore_coverpoint_get_weight(void *cg, int32_t cp_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return 1;
  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return 1;
  auto it = coverpointOptions.find(cp);
  return it != coverpointOptions.end() ? it->second.weight : 1;
}

extern "C" void __moore_coverpoint_set_goal(void *cg, int32_t cp_index,
                                             double goal) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;
  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return;
  if (goal < 0.0)
    goal = 0.0;
  if (goal > 100.0)
    goal = 100.0;
  coverpointOptions[cp].goal = goal;
}

extern "C" double __moore_coverpoint_get_goal(void *cg, int32_t cp_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return 100.0;
  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return 100.0;
  auto it = coverpointOptions.find(cp);
  return it != coverpointOptions.end() ? it->second.goal : 100.0;
}

extern "C" void __moore_coverpoint_set_at_least(void *cg, int32_t cp_index,
                                                 int64_t atLeast) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;
  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return;
  coverpointOptions[cp].atLeast = atLeast > 0 ? atLeast : 1;
}

extern "C" int64_t __moore_coverpoint_get_at_least(void *cg, int32_t cp_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return 1;
  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return 1;
  auto it = coverpointOptions.find(cp);
  return it != coverpointOptions.end() ? it->second.atLeast : 1;
}

extern "C" void __moore_coverpoint_set_comment(void *cg, int32_t cp_index,
                                                const char *comment) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;
  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return;
  coverpointOptions[cp].comment = comment ? comment : "";
}

extern "C" const char *__moore_coverpoint_get_comment(void *cg,
                                                       int32_t cp_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return nullptr;
  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return nullptr;
  auto it = coverpointOptions.find(cp);
  if (it != coverpointOptions.end() && !it->second.comment.empty())
    return it->second.comment.c_str();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Auto Bin Max Options
//===----------------------------------------------------------------------===//

extern "C" void __moore_covergroup_set_auto_bin_max(void *cg, int64_t maxBins) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;
  // auto_bin_max must be positive; IEEE 1800-2017 default is 64
  covergroupOptions[covergroup].autoBinMax = maxBins > 0 ? maxBins : 64;
}

extern "C" int64_t __moore_covergroup_get_auto_bin_max(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return 64;
  auto it = covergroupOptions.find(covergroup);
  return it != covergroupOptions.end() ? it->second.autoBinMax : 64;
}

extern "C" void __moore_coverpoint_set_auto_bin_max(void *cg, int32_t cp_index,
                                                     int64_t maxBins) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;
  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return;
  // auto_bin_max must be positive; IEEE 1800-2017 default is 64
  coverpointOptions[cp].autoBinMax = maxBins > 0 ? maxBins : 64;
}

extern "C" int64_t __moore_coverpoint_get_auto_bin_max(void *cg,
                                                        int32_t cp_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return 64;
  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return 64;
  auto it = coverpointOptions.find(cp);
  return it != coverpointOptions.end() ? it->second.autoBinMax : 64;
}

//===----------------------------------------------------------------------===//
// Generic Coverage Option API
//===----------------------------------------------------------------------===//

extern "C" void __moore_covergroup_set_option(void *cg, int32_t option,
                                               double value) {
  switch (option) {
  case MOORE_OPTION_GOAL:
    __moore_covergroup_set_goal(cg, value);
    break;
  case MOORE_OPTION_WEIGHT:
    __moore_covergroup_set_weight(cg, static_cast<int64_t>(value));
    break;
  case MOORE_OPTION_AT_LEAST:
    __moore_covergroup_set_at_least(cg, static_cast<int64_t>(value));
    break;
  case MOORE_OPTION_AUTO_BIN_MAX:
    __moore_covergroup_set_auto_bin_max(cg, static_cast<int64_t>(value));
    break;
  default:
    // Unknown option - ignore
    break;
  }
}

extern "C" double __moore_covergroup_get_option(void *cg, int32_t option) {
  switch (option) {
  case MOORE_OPTION_GOAL:
    return __moore_covergroup_get_goal(cg);
  case MOORE_OPTION_WEIGHT:
    return static_cast<double>(__moore_covergroup_get_weight(cg));
  case MOORE_OPTION_AT_LEAST:
    return static_cast<double>(__moore_covergroup_get_at_least(cg));
  case MOORE_OPTION_AUTO_BIN_MAX:
    return static_cast<double>(__moore_covergroup_get_auto_bin_max(cg));
  default:
    return 0.0;
  }
}

extern "C" void __moore_coverpoint_set_option(void *cg, int32_t cp_index,
                                               int32_t option, double value) {
  switch (option) {
  case MOORE_OPTION_GOAL:
    __moore_coverpoint_set_goal(cg, cp_index, value);
    break;
  case MOORE_OPTION_WEIGHT:
    __moore_coverpoint_set_weight(cg, cp_index, static_cast<int64_t>(value));
    break;
  case MOORE_OPTION_AT_LEAST:
    __moore_coverpoint_set_at_least(cg, cp_index, static_cast<int64_t>(value));
    break;
  case MOORE_OPTION_AUTO_BIN_MAX:
    __moore_coverpoint_set_auto_bin_max(cg, cp_index,
                                         static_cast<int64_t>(value));
    break;
  default:
    // Unknown option - ignore
    break;
  }
}

extern "C" double __moore_coverpoint_get_option(void *cg, int32_t cp_index,
                                                 int32_t option) {
  switch (option) {
  case MOORE_OPTION_GOAL:
    return __moore_coverpoint_get_goal(cg, cp_index);
  case MOORE_OPTION_WEIGHT:
    return static_cast<double>(__moore_coverpoint_get_weight(cg, cp_index));
  case MOORE_OPTION_AT_LEAST:
    return static_cast<double>(__moore_coverpoint_get_at_least(cg, cp_index));
  case MOORE_OPTION_AUTO_BIN_MAX:
    return static_cast<double>(__moore_coverpoint_get_auto_bin_max(cg, cp_index));
  default:
    return 0.0;
  }
}

//===----------------------------------------------------------------------===//
// Coverage Exclusion APIs
//===----------------------------------------------------------------------===//
//
// Coverage exclusions allow users to mark certain bins as excluded from
// coverage goals. This is useful for unreachable code paths, known limitations,
// or bins that should not be considered for sign-off criteria.
//
// Exclusions differ from ignore_bins in that:
// - ignore_bins are defined in the covergroup specification
// - exclusions are applied at runtime, typically from exclusion files
// - exclusions can be dynamically added/removed during simulation
//

extern "C" void __moore_coverpoint_exclude_bin(void *cg, int32_t cp_index,
                                                const char *bin_name) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp || !bin_name)
    return;

  // Add bin name to excluded set
  excludedBins[cp].insert(bin_name);
}

extern "C" void __moore_coverpoint_include_bin(void *cg, int32_t cp_index,
                                                const char *bin_name) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp || !bin_name)
    return;

  // Remove bin name from excluded set
  auto it = excludedBins.find(cp);
  if (it != excludedBins.end()) {
    it->second.erase(bin_name);
  }
}

extern "C" bool __moore_coverpoint_is_bin_excluded(void *cg, int32_t cp_index,
                                                    const char *bin_name) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return false;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp || !bin_name)
    return false;

  auto it = excludedBins.find(cp);
  if (it != excludedBins.end()) {
    return it->second.count(bin_name) > 0;
  }
  return false;
}

namespace {

/// Parse an exclusion file and apply exclusions.
/// File format (simple text-based):
///   # Comment lines start with #
///   # Empty lines are ignored
///   # Format: covergroup_name.coverpoint_name.bin_name
///   # Wildcards: * matches any sequence of characters
///   cg_name.cp_name.bin_name
///   cg_name.cp_name.*        # Exclude all bins in coverpoint
///   cg_name.*.bin_name       # Exclude bin in all coverpoints
///   *.*.excluded_bin         # Exclude bin in all covergroups/coverpoints
///
bool parseExclusionFile(const char *filename) {
  if (!filename)
    return false;

  FILE *file = std::fopen(filename, "r");
  if (!file)
    return false;

  char line[1024];
  while (std::fgets(line, sizeof(line), file)) {
    // Skip empty lines and comments
    char *start = line;
    while (*start && std::isspace(static_cast<unsigned char>(*start)))
      ++start;
    if (!*start || *start == '#')
      continue;

    // Remove trailing whitespace/newline
    char *end = start + std::strlen(start) - 1;
    while (end > start && std::isspace(static_cast<unsigned char>(*end)))
      *end-- = '\0';

    // Parse format: covergroup.coverpoint.bin
    std::string entry(start);
    size_t firstDot = entry.find('.');
    size_t lastDot = entry.rfind('.');

    if (firstDot == std::string::npos || lastDot == std::string::npos ||
        firstDot == lastDot) {
      // Invalid format - skip
      continue;
    }

    std::string cgPattern = entry.substr(0, firstDot);
    std::string cpPattern = entry.substr(firstDot + 1, lastDot - firstDot - 1);
    std::string binPattern = entry.substr(lastDot + 1);

    // Apply exclusion to matching covergroups/coverpoints/bins
    for (auto *cg : registeredCovergroups) {
      if (!cg || !cg->name)
        continue;

      // Check covergroup name match (simple wildcard: * matches all)
      bool cgMatch = (cgPattern == "*" || cgPattern == cg->name);
      if (!cgMatch)
        continue;

      for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
        auto *cp = cg->coverpoints[i];
        if (!cp || !cp->name)
          continue;

        // Check coverpoint name match
        bool cpMatch = (cpPattern == "*" || cpPattern == cp->name);
        if (!cpMatch)
          continue;

        // Check explicit bins
        auto binDataIt = explicitBinData.find(cp);
        if (binDataIt != explicitBinData.end()) {
          for (const auto &bin : binDataIt->second.bins) {
            if (!bin.name)
              continue;

            // Check bin name match
            bool binMatch = (binPattern == "*" || binPattern == bin.name);
            if (binMatch) {
              excludedBins[cp].insert(bin.name);
            }
          }
        }
      }
    }
  }

  std::fclose(file);
  return true;
}

} // anonymous namespace

extern "C" bool __moore_covergroup_set_exclusion_file(const char *filename) {
  if (!filename)
    return false;

  globalExclusionFile = filename;
  return parseExclusionFile(filename);
}

extern "C" const char *__moore_covergroup_get_exclusion_file(void) {
  return globalExclusionFile.empty() ? nullptr : globalExclusionFile.c_str();
}

extern "C" int32_t __moore_coverpoint_get_excluded_bin_count(void *cg,
                                                              int32_t cp_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return 0;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return 0;

  auto it = excludedBins.find(cp);
  if (it != excludedBins.end()) {
    return static_cast<int32_t>(it->second.size());
  }
  return 0;
}

extern "C" void __moore_coverpoint_clear_exclusions(void *cg, int32_t cp_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return;

  excludedBins.erase(cp);
}

//===----------------------------------------------------------------------===//
// Weighted Coverage Calculation
//===----------------------------------------------------------------------===//

extern "C" double __moore_covergroup_get_weighted_coverage(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || covergroup->num_coverpoints == 0)
    return 0.0;

  double weightedSum = 0.0;
  int64_t totalWeight = 0;

  // Calculate weighted sum of coverpoint coverages
  for (int32_t i = 0; i < covergroup->num_coverpoints; ++i) {
    auto *cp = covergroup->coverpoints[i];
    if (!cp)
      continue;

    int64_t weight = __moore_coverpoint_get_weight(cg, i);
    double coverage = __moore_coverpoint_get_coverage(cg, i);

    weightedSum += coverage * weight;
    totalWeight += weight;
  }

  // Add cross coverage with weights (if implemented)
  auto crossIt = crossCoverageData.find(covergroup);
  if (crossIt != crossCoverageData.end()) {
    for (size_t i = 0; i < crossIt->second.crosses.size(); ++i) {
      // Cross coverage uses weight 1 by default
      double crossCov = __moore_cross_get_coverage(cg, static_cast<int32_t>(i));
      weightedSum += crossCov;
      totalWeight += 1;
    }
  }

  if (totalWeight == 0)
    return 0.0;

  return weightedSum / totalWeight;
}

extern "C" bool __moore_coverpoint_bin_covered(void *cg, int32_t cp_index,
                                                int32_t bin_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return false;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp || !cp->bins || bin_index < 0 || bin_index >= cp->num_bins)
    return false;

  // Get the at_least threshold (coverpoint level overrides covergroup level)
  int64_t atLeast = __moore_coverpoint_get_at_least(cg, cp_index);
  if (atLeast <= 0) {
    // Fall back to covergroup-level at_least
    atLeast = __moore_covergroup_get_at_least(cg);
  }

  return cp->bins[bin_index] >= atLeast;
}

//===----------------------------------------------------------------------===//
// Illegal Bins and Ignore Bins Runtime Support
//===----------------------------------------------------------------------===//

namespace {

/// Structure to store illegal and ignore bins for a coverpoint.
struct SpecialBinData {
  std::vector<MooreCoverageBin> illegalBins;
  std::vector<MooreCoverageBin> ignoreBins;
};

/// Map from coverpoint to its special bin data (illegal/ignore).
thread_local std::map<MooreCoverpoint *, SpecialBinData> specialBinData;

/// Global state for illegal bin handling.
struct IllegalBinState {
  MooreIllegalBinCallback callback = nullptr;
  void *callbackUserData = nullptr;
  bool fatal = true;  // Default: illegal bins cause fatal errors
  int64_t hitCount = 0;
};

thread_local IllegalBinState illegalBinState;

/// Check if a value matches a bin (works for illegal/ignore bins).
bool valueMatchesBin(const MooreCoverageBin &bin, int64_t value) {
  switch (bin.type) {
  case MOORE_BIN_VALUE:
    return value == bin.low;
  case MOORE_BIN_RANGE:
    return value >= bin.low && value <= bin.high;
  case MOORE_BIN_WILDCARD:
    // Wildcard matching uses mask+value encoding:
    // - bin.low = pattern value (don't care bits are 0)
    // - bin.high = mask (1 = don't care, 0 = must match)
    // Match condition: all "care" bits must match the pattern
    // Formula: (value ^ pattern) & ~mask == 0
    return ((value ^ bin.low) & ~bin.high) == 0;
  case MOORE_BIN_TRANSITION:
    // Transitions are not applicable to illegal/ignore value bins
    return false;
  default:
    return false;
  }
}

/// Handle an illegal bin hit.
void handleIllegalBinHit(MooreCovergroup *cg, MooreCoverpoint *cp,
                         const MooreCoverageBin &bin, int64_t value) {
  illegalBinState.hitCount++;

  // Get names for reporting
  const char *cgName = cg ? cg->name : "(unknown)";
  const char *cpName = cp ? cp->name : "(unknown)";
  const char *binName = bin.name ? bin.name : "(unnamed)";

  // Call user callback if registered
  if (illegalBinState.callback) {
    illegalBinState.callback(cgName, cpName, binName, value,
                             illegalBinState.callbackUserData);
  }

  // Print error/warning message
  if (illegalBinState.fatal) {
    std::fprintf(stderr,
                 "Error: Illegal bin hit in covergroup '%s', coverpoint '%s', "
                 "bin '%s': value = %ld\n",
                 cgName, cpName, binName, static_cast<long>(value));
    // In a real simulator, this would call $fatal. For now, we print and continue.
    // The caller can check the hit count and terminate if desired.
  } else {
    std::fprintf(stderr,
                 "Warning: Illegal bin hit in covergroup '%s', coverpoint '%s', "
                 "bin '%s': value = %ld\n",
                 cgName, cpName, binName, static_cast<long>(value));
  }
}

/// Check if a value matches any illegal bins and handle accordingly.
/// Returns true if an illegal bin was hit.
bool checkIllegalBinsInternal(MooreCovergroup *cg, MooreCoverpoint *cp,
                              int64_t value) {
  auto it = specialBinData.find(cp);
  if (it == specialBinData.end())
    return false;

  for (auto &bin : it->second.illegalBins) {
    if (valueMatchesBin(bin, value)) {
      bin.hit_count++;
      handleIllegalBinHit(cg, cp, bin, value);
      return true;
    }
  }

  return false;
}

/// Check if a value matches any ignore bins.
/// Returns true if the value should be ignored.
bool checkIgnoreBinsInternal(MooreCoverpoint *cp, int64_t value) {
  auto it = specialBinData.find(cp);
  if (it == specialBinData.end())
    return false;

  for (const auto &bin : it->second.ignoreBins) {
    if (valueMatchesBin(bin, value))
      return true;
  }

  return false;
}

} // anonymous namespace

extern "C" void __moore_coverpoint_set_illegal_bins(void *cg, int32_t cp_index,
                                                     MooreCoverageBin *bins,
                                                     int32_t num_bins) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return;

  // Clear existing illegal bins and add new ones
  auto &data = specialBinData[cp];
  data.illegalBins.clear();

  if (bins && num_bins > 0) {
    for (int32_t i = 0; i < num_bins; ++i) {
      MooreCoverageBin bin = bins[i];
      bin.kind = MOORE_BIN_KIND_ILLEGAL;
      bin.hit_count = 0;
      data.illegalBins.push_back(bin);
    }
  }
}

extern "C" void __moore_coverpoint_set_ignore_bins(void *cg, int32_t cp_index,
                                                    MooreCoverageBin *bins,
                                                    int32_t num_bins) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return;

  // Clear existing ignore bins and add new ones
  auto &data = specialBinData[cp];
  data.ignoreBins.clear();

  if (bins && num_bins > 0) {
    for (int32_t i = 0; i < num_bins; ++i) {
      MooreCoverageBin bin = bins[i];
      bin.kind = MOORE_BIN_KIND_IGNORE;
      bin.hit_count = 0;
      data.ignoreBins.push_back(bin);
    }
  }
}

extern "C" void __moore_coverpoint_add_illegal_bin(void *cg, int32_t cp_index,
                                                    const char *bin_name,
                                                    int64_t low, int64_t high) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return;

  MooreCoverageBin bin;
  bin.name = bin_name;
  bin.type = (low == high) ? MOORE_BIN_VALUE : MOORE_BIN_RANGE;
  bin.kind = MOORE_BIN_KIND_ILLEGAL;
  bin.low = low;
  bin.high = high;
  bin.hit_count = 0;

  specialBinData[cp].illegalBins.push_back(bin);
}

extern "C" void __moore_coverpoint_add_ignore_bin(void *cg, int32_t cp_index,
                                                   const char *bin_name,
                                                   int64_t low, int64_t high) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return;

  MooreCoverageBin bin;
  bin.name = bin_name;
  bin.type = (low == high) ? MOORE_BIN_VALUE : MOORE_BIN_RANGE;
  bin.kind = MOORE_BIN_KIND_IGNORE;
  bin.low = low;
  bin.high = high;
  bin.hit_count = 0;

  specialBinData[cp].ignoreBins.push_back(bin);
}

extern "C" void
__moore_coverage_set_illegal_bin_callback(MooreIllegalBinCallback callback,
                                          void *userData) {
  illegalBinState.callback = callback;
  illegalBinState.callbackUserData = userData;
}

extern "C" void __moore_coverage_set_illegal_bin_fatal(bool fatal) {
  illegalBinState.fatal = fatal;
}

extern "C" bool __moore_coverage_illegal_bin_is_fatal(void) {
  return illegalBinState.fatal;
}

extern "C" int64_t __moore_coverage_get_illegal_bin_hits(void) {
  return illegalBinState.hitCount;
}

extern "C" void __moore_coverage_reset_illegal_bin_hits(void) {
  illegalBinState.hitCount = 0;
}

extern "C" bool __moore_coverpoint_is_ignored(void *cg, int32_t cp_index,
                                               int64_t value) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return false;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return false;

  auto it = specialBinData.find(cp);
  if (it == specialBinData.end())
    return false;

  for (const auto &bin : it->second.ignoreBins) {
    if (valueMatchesBin(bin, value))
      return true;
  }

  return false;
}

extern "C" bool __moore_coverpoint_is_illegal(void *cg, int32_t cp_index,
                                               int64_t value) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return false;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return false;

  auto it = specialBinData.find(cp);
  if (it == specialBinData.end())
    return false;

  for (const auto &bin : it->second.illegalBins) {
    if (valueMatchesBin(bin, value))
      return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Coverage Sample Callbacks and Explicit Sample API
//===----------------------------------------------------------------------===//
//
// These functions implement explicit sample() method and callback support for
// covergroups, as specified in IEEE 1800-2017 Section 19.8.
//
// SystemVerilog supports:
// - Explicit sample() method: cg.sample();
// - Sample with arguments: cg.sample(val1, val2);
// - Sample events: covergroup cg @(posedge clk);
// - pre_sample() and post_sample() callbacks
//

namespace {

/// Sample callback function signature.
/// @param cg Pointer to the covergroup being sampled
/// @param args Array of sample arguments (NULL if no arguments)
/// @param num_args Number of sample arguments (0 if no arguments)
/// @param userData User-provided context data
using MooreSampleCallback = void (*)(void *cg, int64_t *args, int32_t num_args,
                                      void *userData);

/// Structure to store sample callback data for a covergroup.
struct CovergroupSampleCallbacks {
  MooreSampleCallback preSampleCallback = nullptr;
  void *preSampleUserData = nullptr;
  MooreSampleCallback postSampleCallback = nullptr;
  void *postSampleUserData = nullptr;

  // Sample argument mapping: which coverpoint gets which sample argument
  // Index i = coverpoint index, value = sample argument index (-1 = none)
  std::vector<int32_t> sampleArgMapping;

  // Enable/disable flag for sampling
  bool enabled = true;

  // Sample event trigger (for @(event) syntax support)
  bool sampleEventEnabled = false;
  const char *sampleEventName = nullptr;
};

/// Map from covergroup to its sample callback data.
thread_local std::map<MooreCovergroup *, CovergroupSampleCallbacks>
    covergroupSampleCallbacks;

/// Global pre/post sample callbacks (apply to all covergroups).
thread_local MooreSampleCallback globalPreSampleCallback = nullptr;
thread_local void *globalPreSampleUserData = nullptr;
thread_local MooreSampleCallback globalPostSampleCallback = nullptr;
thread_local void *globalPostSampleUserData = nullptr;

} // anonymous namespace

extern "C" void __moore_covergroup_sample(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;

  // Check if sampling is enabled
  auto callbackIt = covergroupSampleCallbacks.find(covergroup);
  if (callbackIt != covergroupSampleCallbacks.end() &&
      !callbackIt->second.enabled)
    return;

  // Invoke global pre-sample callback
  if (globalPreSampleCallback) {
    globalPreSampleCallback(cg, nullptr, 0, globalPreSampleUserData);
  }

  // Invoke covergroup-specific pre-sample callback
  if (callbackIt != covergroupSampleCallbacks.end() &&
      callbackIt->second.preSampleCallback) {
    callbackIt->second.preSampleCallback(cg, nullptr, 0,
                                         callbackIt->second.preSampleUserData);
  }

  // For a basic sample() call with no arguments, we don't sample any
  // coverpoints directly - the user is expected to have set up sample_event
  // triggers or explicit coverpoint sample calls elsewhere.
  // This function is primarily for triggering callbacks.

  // Invoke covergroup-specific post-sample callback
  if (callbackIt != covergroupSampleCallbacks.end() &&
      callbackIt->second.postSampleCallback) {
    callbackIt->second.postSampleCallback(cg, nullptr, 0,
                                          callbackIt->second.postSampleUserData);
  }

  // Invoke global post-sample callback
  if (globalPostSampleCallback) {
    globalPostSampleCallback(cg, nullptr, 0, globalPostSampleUserData);
  }
}

extern "C" void __moore_covergroup_sample_with_args(void *cg, int64_t *args,
                                                     int32_t num_args) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;

  // Check if sampling is enabled
  auto callbackIt = covergroupSampleCallbacks.find(covergroup);
  if (callbackIt != covergroupSampleCallbacks.end() &&
      !callbackIt->second.enabled)
    return;

  // Invoke global pre-sample callback
  if (globalPreSampleCallback) {
    globalPreSampleCallback(cg, args, num_args, globalPreSampleUserData);
  }

  // Invoke covergroup-specific pre-sample callback
  if (callbackIt != covergroupSampleCallbacks.end() &&
      callbackIt->second.preSampleCallback) {
    callbackIt->second.preSampleCallback(cg, args, num_args,
                                         callbackIt->second.preSampleUserData);
  }

  // Sample coverpoints based on argument mapping
  if (callbackIt != covergroupSampleCallbacks.end() &&
      !callbackIt->second.sampleArgMapping.empty()) {
    for (int32_t cpIdx = 0;
         cpIdx < static_cast<int32_t>(callbackIt->second.sampleArgMapping.size());
         ++cpIdx) {
      int32_t argIdx = callbackIt->second.sampleArgMapping[cpIdx];
      if (argIdx >= 0 && argIdx < num_args) {
        __moore_coverpoint_sample(cg, cpIdx, args[argIdx]);
      }
    }
  } else if (args && num_args > 0) {
    // Default behavior: sample first num_args coverpoints with corresponding args
    int32_t maxIdx = std::min(num_args, covergroup->num_coverpoints);
    for (int32_t i = 0; i < maxIdx; ++i) {
      __moore_coverpoint_sample(cg, i, args[i]);
    }
  }

  // Invoke covergroup-specific post-sample callback
  if (callbackIt != covergroupSampleCallbacks.end() &&
      callbackIt->second.postSampleCallback) {
    callbackIt->second.postSampleCallback(cg, args, num_args,
                                          callbackIt->second.postSampleUserData);
  }

  // Invoke global post-sample callback
  if (globalPostSampleCallback) {
    globalPostSampleCallback(cg, args, num_args, globalPostSampleUserData);
  }
}

extern "C" void __moore_covergroup_set_pre_sample_callback(
    void *cg, void (*callback)(void *, int64_t *, int32_t, void *),
    void *userData) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;

  covergroupSampleCallbacks[covergroup].preSampleCallback = callback;
  covergroupSampleCallbacks[covergroup].preSampleUserData = userData;
}

extern "C" void __moore_covergroup_set_post_sample_callback(
    void *cg, void (*callback)(void *, int64_t *, int32_t, void *),
    void *userData) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;

  covergroupSampleCallbacks[covergroup].postSampleCallback = callback;
  covergroupSampleCallbacks[covergroup].postSampleUserData = userData;
}

extern "C" void __moore_coverage_set_global_pre_sample_callback(
    void (*callback)(void *, int64_t *, int32_t, void *), void *userData) {
  globalPreSampleCallback = callback;
  globalPreSampleUserData = userData;
}

extern "C" void __moore_coverage_set_global_post_sample_callback(
    void (*callback)(void *, int64_t *, int32_t, void *), void *userData) {
  globalPostSampleCallback = callback;
  globalPostSampleUserData = userData;
}

extern "C" void __moore_covergroup_set_sample_arg_mapping(void *cg,
                                                           int32_t *mapping,
                                                           int32_t num_mappings) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;

  auto &callbacks = covergroupSampleCallbacks[covergroup];
  callbacks.sampleArgMapping.clear();
  if (mapping && num_mappings > 0) {
    callbacks.sampleArgMapping.assign(mapping, mapping + num_mappings);
  }
}

extern "C" void __moore_covergroup_set_sample_enabled(void *cg, bool enabled) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;

  covergroupSampleCallbacks[covergroup].enabled = enabled;
}

extern "C" bool __moore_covergroup_is_sample_enabled(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return false;

  auto it = covergroupSampleCallbacks.find(covergroup);
  if (it == covergroupSampleCallbacks.end())
    return true; // Default is enabled

  return it->second.enabled;
}

extern "C" void __moore_covergroup_set_sample_event(void *cg,
                                                     const char *eventName) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;

  auto &callbacks = covergroupSampleCallbacks[covergroup];
  callbacks.sampleEventEnabled = (eventName != nullptr);
  callbacks.sampleEventName = eventName;
}

extern "C" const char *__moore_covergroup_get_sample_event(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return nullptr;

  auto it = covergroupSampleCallbacks.find(covergroup);
  if (it == covergroupSampleCallbacks.end())
    return nullptr;

  return it->second.sampleEventName;
}

extern "C" bool __moore_covergroup_has_sample_event(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return false;

  auto it = covergroupSampleCallbacks.find(covergroup);
  if (it == covergroupSampleCallbacks.end())
    return false;

  return it->second.sampleEventEnabled;
}

extern "C" void __moore_covergroup_trigger_sample_event(void *cg,
                                                         const char *eventName) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup)
    return;

  auto it = covergroupSampleCallbacks.find(covergroup);
  if (it == covergroupSampleCallbacks.end())
    return;

  // Check if this event matches the configured sample event
  if (!it->second.sampleEventEnabled)
    return;

  // If eventName is NULL, always trigger; otherwise check for match
  if (eventName && it->second.sampleEventName) {
    if (std::strcmp(eventName, it->second.sampleEventName) != 0)
      return;
  }

  // Trigger the sample
  __moore_covergroup_sample(cg);
}

// Cleanup function called from __moore_covergroup_destroy
void __moore_sample_callbacks_cleanup_for_covergroup(MooreCovergroup *cg) {
  covergroupSampleCallbacks.erase(cg);
}

//===----------------------------------------------------------------------===//
// Coverage Exclusion API
//===----------------------------------------------------------------------===//

namespace {

/// Structure to store an exclusion pattern.
struct ExclusionPattern {
  std::string pattern;
  std::string cgPattern;  // Covergroup pattern (first component)
  std::string cpPattern;  // Coverpoint pattern (second component)
  std::string binPattern; // Bin pattern (third component)
};

/// Global list of exclusion patterns.
thread_local std::vector<ExclusionPattern> exclusionPatterns;

/// Parse an exclusion pattern into components.
/// Format: "covergroup.coverpoint.bin" (with wildcards)
ExclusionPattern parseExclusionPattern(const std::string &pattern) {
  ExclusionPattern result;
  result.pattern = pattern;

  // Split by '.'
  size_t pos1 = pattern.find('.');
  if (pos1 == std::string::npos) {
    // Only covergroup specified, wildcard the rest
    result.cgPattern = pattern;
    result.cpPattern = "*";
    result.binPattern = "*";
  } else {
    result.cgPattern = pattern.substr(0, pos1);
    size_t pos2 = pattern.find('.', pos1 + 1);
    if (pos2 == std::string::npos) {
      // Covergroup.coverpoint specified
      result.cpPattern = pattern.substr(pos1 + 1);
      result.binPattern = "*";
    } else {
      // Full pattern
      result.cpPattern = pattern.substr(pos1 + 1, pos2 - pos1 - 1);
      result.binPattern = pattern.substr(pos2 + 1);
    }
  }

  return result;
}

/// Check if a name matches a wildcard pattern.
/// Supports '*' (any sequence) and '?' (single character).
bool matchesWildcard(const std::string &pattern, const std::string &name) {
  size_t p = 0, n = 0;
  size_t starP = std::string::npos, starN = 0;

  while (n < name.size()) {
    if (p < pattern.size() && (pattern[p] == name[n] || pattern[p] == '?')) {
      ++p;
      ++n;
    } else if (p < pattern.size() && pattern[p] == '*') {
      starP = p;
      starN = n;
      ++p;
    } else if (starP != std::string::npos) {
      p = starP + 1;
      ++starN;
      n = starN;
    } else {
      return false;
    }
  }

  // Skip trailing wildcards
  while (p < pattern.size() && pattern[p] == '*')
    ++p;

  return p == pattern.size();
}

/// Check if a covergroup/coverpoint/bin matches an exclusion pattern.
bool matchesExclusion(const ExclusionPattern &excl, const char *cgName,
                      const char *cpName, const char *binName) {
  std::string cg = cgName ? cgName : "";
  std::string cp = cpName ? cpName : "";
  std::string bin = binName ? binName : "";

  return matchesWildcard(excl.cgPattern, cg) &&
         matchesWildcard(excl.cpPattern, cp) &&
         matchesWildcard(excl.binPattern, bin);
}

} // anonymous namespace

extern "C" int32_t __moore_coverage_add_exclusion(const char *pattern) {
  if (!pattern || !*pattern)
    return 1;

  ExclusionPattern excl = parseExclusionPattern(pattern);
  exclusionPatterns.push_back(excl);
  return 0;
}

extern "C" int32_t __moore_coverage_remove_exclusion(const char *pattern) {
  if (!pattern)
    return 1;

  std::string patternStr(pattern);
  for (auto it = exclusionPatterns.begin(); it != exclusionPatterns.end(); ++it) {
    if (it->pattern == patternStr) {
      exclusionPatterns.erase(it);
      return 0;
    }
  }
  return 1; // Pattern not found
}

extern "C" void __moore_coverage_clear_exclusions(void) {
  exclusionPatterns.clear();
}

extern "C" int32_t __moore_coverage_load_exclusions(const char *filename) {
  if (!filename)
    return -1;

  FILE *fp = std::fopen(filename, "r");
  if (!fp)
    return -1;

  int32_t count = 0;
  char line[1024];

  while (std::fgets(line, sizeof(line), fp)) {
    // Skip comments and empty lines
    char *start = line;
    while (*start == ' ' || *start == '\t')
      ++start;

    if (*start == '#' || *start == '\n' || *start == '\0')
      continue;

    // Remove trailing newline
    size_t len = std::strlen(start);
    if (len > 0 && start[len - 1] == '\n')
      start[len - 1] = '\0';

    // Trim trailing whitespace
    len = std::strlen(start);
    while (len > 0 && (start[len - 1] == ' ' || start[len - 1] == '\t')) {
      start[len - 1] = '\0';
      --len;
    }

    if (len > 0 && __moore_coverage_add_exclusion(start) == 0)
      ++count;
  }

  std::fclose(fp);
  return count;
}

extern "C" int32_t __moore_coverage_save_exclusions(const char *filename) {
  if (!filename)
    return 1;

  FILE *fp = std::fopen(filename, "w");
  if (!fp)
    return 1;

  std::fprintf(fp, "# CIRCT Coverage Exclusion File\n");
  std::fprintf(fp, "# Format: covergroup.coverpoint.bin\n");
  std::fprintf(fp, "# Wildcards: * (any sequence), ? (single character)\n\n");

  for (const auto &excl : exclusionPatterns) {
    std::fprintf(fp, "%s\n", excl.pattern.c_str());
  }

  std::fclose(fp);
  return 0;
}

extern "C" bool __moore_coverage_is_excluded(const char *cg_name,
                                              const char *cp_name,
                                              const char *bin_name) {
  for (const auto &excl : exclusionPatterns) {
    if (matchesExclusion(excl, cg_name, cp_name, bin_name))
      return true;
  }
  return false;
}

extern "C" int32_t __moore_coverage_get_exclusion_count(void) {
  return static_cast<int32_t>(exclusionPatterns.size());
}

//===----------------------------------------------------------------------===//
// HTML Coverage Report
//===----------------------------------------------------------------------===//

extern "C" int32_t __moore_coverage_report_html(const char *filename) {
  if (!filename)
    return 1;

  FILE *fp = std::fopen(filename, "w");
  if (!fp)
    return 1;

  // Generate HTML report with embedded CSS and JavaScript
  std::string html;
  html += "<!DOCTYPE html>\n";
  html += "<html lang=\"en\">\n";
  html += "<head>\n";
  html += "  <meta charset=\"UTF-8\">\n";
  html += "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n";
  html += "  <title>CIRCT Coverage Report</title>\n";
  html += "  <style>\n";
  html += "    :root {\n";
  html += "      --bg-primary: #1a1a2e;\n";
  html += "      --bg-secondary: #16213e;\n";
  html += "      --text-primary: #eee;\n";
  html += "      --text-secondary: #aaa;\n";
  html += "      --accent: #0f4c75;\n";
  html += "      --success: #28a745;\n";
  html += "      --warning: #ffc107;\n";
  html += "      --danger: #dc3545;\n";
  html += "    }\n";
  html += "    body {\n";
  html += "      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;\n";
  html += "      background: var(--bg-primary);\n";
  html += "      color: var(--text-primary);\n";
  html += "      margin: 0;\n";
  html += "      padding: 20px;\n";
  html += "    }\n";
  html += "    h1, h2, h3 { margin-top: 0; }\n";
  html += "    .container { max-width: 1200px; margin: 0 auto; }\n";
  html += "    .header {\n";
  html += "      background: var(--bg-secondary);\n";
  html += "      padding: 20px;\n";
  html += "      border-radius: 8px;\n";
  html += "      margin-bottom: 20px;\n";
  html += "    }\n";
  html += "    .summary {\n";
  html += "      display: grid;\n";
  html += "      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));\n";
  html += "      gap: 15px;\n";
  html += "      margin-bottom: 20px;\n";
  html += "    }\n";
  html += "    .stat-card {\n";
  html += "      background: var(--bg-secondary);\n";
  html += "      padding: 15px;\n";
  html += "      border-radius: 8px;\n";
  html += "      text-align: center;\n";
  html += "    }\n";
  html += "    .stat-value {\n";
  html += "      font-size: 2em;\n";
  html += "      font-weight: bold;\n";
  html += "    }\n";
  html += "    .stat-label {\n";
  html += "      color: var(--text-secondary);\n";
  html += "      font-size: 0.9em;\n";
  html += "    }\n";
  html += "    .covergroup {\n";
  html += "      background: var(--bg-secondary);\n";
  html += "      border-radius: 8px;\n";
  html += "      padding: 20px;\n";
  html += "      margin-bottom: 15px;\n";
  html += "    }\n";
  html += "    .coverpoint {\n";
  html += "      background: rgba(255,255,255,0.05);\n";
  html += "      border-radius: 6px;\n";
  html += "      padding: 15px;\n";
  html += "      margin: 10px 0;\n";
  html += "    }\n";
  html += "    .progress-bar {\n";
  html += "      background: rgba(255,255,255,0.1);\n";
  html += "      border-radius: 4px;\n";
  html += "      height: 20px;\n";
  html += "      overflow: hidden;\n";
  html += "      margin: 5px 0;\n";
  html += "    }\n";
  html += "    .progress-fill {\n";
  html += "      height: 100%;\n";
  html += "      border-radius: 4px;\n";
  html += "      transition: width 0.3s ease;\n";
  html += "    }\n";
  html += "    .coverage-high { background: var(--success); }\n";
  html += "    .coverage-med { background: var(--warning); }\n";
  html += "    .coverage-low { background: var(--danger); }\n";
  html += "    .meta { color: var(--text-secondary); font-size: 0.85em; }\n";
  html += "    table {\n";
  html += "      width: 100%;\n";
  html += "      border-collapse: collapse;\n";
  html += "      margin: 10px 0;\n";
  html += "    }\n";
  html += "    th, td {\n";
  html += "      padding: 8px 12px;\n";
  html += "      text-align: left;\n";
  html += "      border-bottom: 1px solid rgba(255,255,255,0.1);\n";
  html += "    }\n";
  html += "    th { background: rgba(255,255,255,0.05); }\n";
  html += "    th.sortable { cursor: pointer; user-select: none; }\n";
  html += "    th.sortable:hover { background: rgba(255,255,255,0.1); }\n";
  html += "    th.sortable::after { content: ' \\2195'; opacity: 0.5; }\n";
  html += "    .collapsible {\n";
  html += "      cursor: pointer;\n";
  html += "      display: flex;\n";
  html += "      align-items: center;\n";
  html += "      justify-content: space-between;\n";
  html += "    }\n";
  html += "    .collapsible::after {\n";
  html += "      content: '\\25BC';\n";
  html += "      font-size: 0.8em;\n";
  html += "      transition: transform 0.3s ease;\n";
  html += "    }\n";
  html += "    .collapsible.collapsed::after {\n";
  html += "      transform: rotate(-90deg);\n";
  html += "    }\n";
  html += "    .collapse-content {\n";
  html += "      overflow: hidden;\n";
  html += "      transition: max-height 0.3s ease;\n";
  html += "    }\n";
  html += "    .collapse-content.collapsed {\n";
  html += "      max-height: 0 !important;\n";
  html += "    }\n";
  html += "    .filter-bar {\n";
  html += "      margin: 15px 0;\n";
  html += "      display: flex;\n";
  html += "      gap: 15px;\n";
  html += "      flex-wrap: wrap;\n";
  html += "      align-items: center;\n";
  html += "    }\n";
  html += "    .filter-bar input, .filter-bar select {\n";
  html += "      background: var(--bg-secondary);\n";
  html += "      border: 1px solid rgba(255,255,255,0.2);\n";
  html += "      border-radius: 4px;\n";
  html += "      padding: 8px 12px;\n";
  html += "      color: var(--text-primary);\n";
  html += "    }\n";
  html += "    .filter-bar input:focus, .filter-bar select:focus {\n";
  html += "      outline: none;\n";
  html += "      border-color: var(--accent);\n";
  html += "    }\n";
  html += "    .btn {\n";
  html += "      background: var(--accent);\n";
  html += "      border: none;\n";
  html += "      border-radius: 4px;\n";
  html += "      padding: 8px 16px;\n";
  html += "      color: white;\n";
  html += "      cursor: pointer;\n";
  html += "      transition: background 0.2s;\n";
  html += "    }\n";
  html += "    .btn:hover { background: #0d3d5c; }\n";
  html += "    .hidden { display: none !important; }\n";
  html += "    @media print {\n";
  html += "      .filter-bar, .btn, .collapsible::after { display: none; }\n";
  html += "      .collapse-content { max-height: none !important; }\n";
  html += "      body { background: white; color: black; }\n";
  html += "      .covergroup, .stat-card, .header { background: #f5f5f5; }\n";
  html += "    }\n";
  html += "  </style>\n";
  html += "</head>\n";
  html += "<body>\n";
  html += "  <div class=\"container\">\n";
  html += "    <div class=\"header\">\n";
  html += "      <h1>CIRCT Coverage Report</h1>\n";

  // Add timestamp
  std::time_t now = std::time(nullptr);
  char timeBuf[64];
  std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
  html += "      <p class=\"meta\">Generated by circt-moore-runtime on " + std::string(timeBuf) + "</p>\n";
  html += "    </div>\n";

  // Filter bar
  html += "    <div class=\"filter-bar\">\n";
  html += "      <input type=\"text\" id=\"searchInput\" placeholder=\"Search covergroups...\" oninput=\"filterCovergroups()\">\n";
  html += "      <select id=\"statusFilter\" onchange=\"filterCovergroups()\">\n";
  html += "        <option value=\"all\">All Status</option>\n";
  html += "        <option value=\"passed\">Passed (Goal Met)</option>\n";
  html += "        <option value=\"failing\">Failing (Goal Not Met)</option>\n";
  html += "      </select>\n";
  html += "      <select id=\"coverageFilter\" onchange=\"filterCovergroups()\">\n";
  html += "        <option value=\"all\">All Coverage</option>\n";
  html += "        <option value=\"100\">100%</option>\n";
  html += "        <option value=\"high\">High (>=80%)</option>\n";
  html += "        <option value=\"medium\">Medium (50-79%)</option>\n";
  html += "        <option value=\"low\">Low (<50%)</option>\n";
  html += "      </select>\n";
  html += "      <button class=\"btn\" onclick=\"expandAll()\">Expand All</button>\n";
  html += "      <button class=\"btn\" onclick=\"collapseAll()\">Collapse All</button>\n";
  html += "      <button class=\"btn\" onclick=\"window.print()\">Print Report</button>\n";
  html += "    </div>\n";

  // Summary section
  double totalCoverage = __moore_coverage_get_total();
  int32_t numCovergroups = __moore_coverage_get_num_covergroups();
  int32_t totalCoverpoints = 0;
  int64_t totalHits = 0;

  for (auto *cg : registeredCovergroups) {
    if (cg) {
      totalCoverpoints += cg->num_coverpoints;
      for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
        if (cg->coverpoints[i])
          totalHits += cg->coverpoints[i]->hits;
      }
    }
  }

  html += "    <div class=\"summary\">\n";
  html += "      <div class=\"stat-card\">\n";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.1f%%", totalCoverage);
  html += "        <div class=\"stat-value\">" + std::string(buf) + "</div>\n";
  html += "        <div class=\"stat-label\">Total Coverage</div>\n";
  html += "      </div>\n";
  html += "      <div class=\"stat-card\">\n";
  html += "        <div class=\"stat-value\">" + std::to_string(numCovergroups) + "</div>\n";
  html += "        <div class=\"stat-label\">Covergroups</div>\n";
  html += "      </div>\n";
  html += "      <div class=\"stat-card\">\n";
  html += "        <div class=\"stat-value\">" + std::to_string(totalCoverpoints) + "</div>\n";
  html += "        <div class=\"stat-label\">Coverpoints</div>\n";
  html += "      </div>\n";
  html += "      <div class=\"stat-card\">\n";
  html += "        <div class=\"stat-value\">" + std::to_string(totalHits) + "</div>\n";
  html += "        <div class=\"stat-label\">Total Samples</div>\n";
  html += "      </div>\n";
  html += "    </div>\n";

  // Covergroup details
  for (auto *cg : registeredCovergroups) {
    if (!cg)
      continue;

    double cgCoverage = __moore_covergroup_get_coverage(cg);
    double goal = __moore_covergroup_get_goal(cg);
    bool goalMet = cgCoverage >= goal;

    // Determine coverage level for filtering
    std::string coverageLevel = cgCoverage >= 100 ? "100" :
                                (cgCoverage >= 80 ? "high" :
                                (cgCoverage >= 50 ? "medium" : "low"));

    html += "    <div class=\"covergroup\" data-name=\"" +
            std::string(cg->name ? cg->name : "(unnamed)") +
            "\" data-status=\"" + (goalMet ? "passed" : "failing") +
            "\" data-coverage=\"" + coverageLevel + "\">\n";
    html += "      <h2 class=\"collapsible\" onclick=\"toggleCollapse(this)\">" +
            std::string(cg->name ? cg->name : "(unnamed)") + "</h2>\n";
    html += "      <div class=\"collapse-content\">\n";
    html += "      <div class=\"progress-bar\">\n";
    std::string coverageClass = cgCoverage >= 80 ? "coverage-high" :
                                (cgCoverage >= 50 ? "coverage-med" : "coverage-low");
    std::snprintf(buf, sizeof(buf), "%.1f", cgCoverage);
    html += "        <div class=\"progress-fill " + coverageClass +
            "\" style=\"width: " + std::string(buf) + "%\"></div>\n";
    html += "      </div>\n";
    std::snprintf(buf, sizeof(buf), "%.2f%% coverage (goal: %.0f%%)", cgCoverage, goal);
    html += "      <p class=\"meta\">" + std::string(buf) +
            (goalMet ? " - PASSED" : " - NOT MET") + "</p>\n";

    // Coverpoints table
    if (cg->num_coverpoints > 0) {
      html += "      <h3 class=\"collapsible\" onclick=\"toggleCollapse(this)\">Coverpoints</h3>\n";
      html += "      <div class=\"collapse-content\">\n";
      html += "      <table class=\"sortable-table\">\n";
      html += "        <thead>\n";
      html += "          <tr>\n";
      html += "            <th class=\"sortable\" onclick=\"sortTable(this, 0)\">Coverpoint</th>\n";
      html += "            <th class=\"sortable\" onclick=\"sortTable(this, 1)\">Hits</th>\n";
      html += "            <th class=\"sortable\" onclick=\"sortTable(this, 2)\">Unique Values</th>\n";
      html += "            <th>Range</th>\n";
      html += "            <th class=\"sortable\" onclick=\"sortTable(this, 4)\">Coverage</th>\n";
      html += "          </tr>\n";
      html += "        </thead>\n";
      html += "        <tbody>\n";

      for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
        auto *cp = cg->coverpoints[i];
        if (!cp)
          continue;

        double cpCoverage = __moore_coverpoint_get_coverage(cg, i);
        auto trackerIt = coverpointTrackers.find(cp);
        int64_t uniqueVals = 0;
        if (trackerIt != coverpointTrackers.end()) {
          uniqueVals = static_cast<int64_t>(trackerIt->second.valueCounts.size());
        }

        html += "          <tr>\n";
        html += "            <td>" + std::string(cp->name ? cp->name : "(unnamed)") + "</td>\n";
        html += "            <td>" + std::to_string(cp->hits) + "</td>\n";
        html += "            <td>" + std::to_string(uniqueVals) + "</td>\n";

        if (cp->hits > 0 && cp->min_val <= cp->max_val) {
          html += "            <td>" + std::to_string(cp->min_val) + ".." +
                  std::to_string(cp->max_val) + "</td>\n";
        } else {
          html += "            <td>-</td>\n";
        }

        // Color coding: green (100%), yellow (50-99%), red (<50%)
        std::snprintf(buf, sizeof(buf), "%.1f%%", cpCoverage);
        std::string cpColorVar = cpCoverage >= 100.0 ? "success" :
                                 (cpCoverage >= 50.0 ? "warning" : "danger");
        html += "            <td><span style=\"color: var(--" + cpColorVar +
                ")\">" + std::string(buf) + "</span></td>\n";
        html += "          </tr>\n";
      }

      html += "        </tbody>\n";
      html += "      </table>\n";

      // Per-coverpoint bin details
      for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
        auto *cp = cg->coverpoints[i];
        if (!cp)
          continue;

        auto binDataIt = explicitBinData.find(cp);
        if (binDataIt != explicitBinData.end() && !binDataIt->second.bins.empty()) {
          html += "      <div class=\"coverpoint\">\n";
          html += "        <h4>" + std::string(cp->name ? cp->name : "(unnamed)") + " - Bins</h4>\n";
          html += "        <table>\n";
          html += "          <thead>\n";
          html += "            <tr>\n";
          html += "              <th>Bin Name</th>\n";
          html += "              <th>Type</th>\n";
          html += "              <th>Kind</th>\n";
          html += "              <th>Range/Value</th>\n";
          html += "              <th>Hit Count</th>\n";
          html += "            </tr>\n";
          html += "          </thead>\n";
          html += "          <tbody>\n";

          for (size_t binIdx = 0; binIdx < binDataIt->second.bins.size(); ++binIdx) {
            const auto &bin = binDataIt->second.bins[binIdx];
            html += "            <tr>\n";
            html += "              <td>" + std::string(bin.name ? bin.name : "(unnamed)") + "</td>\n";

            // Bin type
            std::string binType;
            switch (bin.type) {
              case MOORE_BIN_VALUE: binType = "value"; break;
              case MOORE_BIN_RANGE: binType = "range"; break;
              case MOORE_BIN_WILDCARD: binType = "wildcard"; break;
              case MOORE_BIN_TRANSITION: binType = "transition"; break;
              default: binType = "unknown"; break;
            }
            html += "              <td>" + binType + "</td>\n";

            // Bin kind
            std::string binKind;
            switch (bin.kind) {
              case MOORE_BIN_KIND_NORMAL: binKind = "normal"; break;
              case MOORE_BIN_KIND_IGNORE: binKind = "ignore"; break;
              case MOORE_BIN_KIND_ILLEGAL: binKind = "illegal"; break;
              default: binKind = "unknown"; break;
            }
            html += "              <td>" + binKind + "</td>\n";

            // Range/value
            if (bin.type == MOORE_BIN_VALUE) {
              html += "              <td>" + std::to_string(bin.low) + "</td>\n";
            } else if (bin.type == MOORE_BIN_RANGE) {
              html += "              <td>" + std::to_string(bin.low) + ".." +
                      std::to_string(bin.high) + "</td>\n";
            } else if (bin.type == MOORE_BIN_WILDCARD) {
              html += "              <td>pattern: " + std::to_string(bin.low) +
                      ", mask: " + std::to_string(bin.high) + "</td>\n";
            } else {
              html += "              <td>-</td>\n";
            }

            // Hit count - use stored bin hit count from coverpoint bins array
            int64_t hitCount = (cp->bins && static_cast<int32_t>(binIdx) < cp->num_bins) ?
                               cp->bins[binIdx] : bin.hit_count;
            std::string hitColor = hitCount > 0 ? "success" : "danger";
            html += "              <td><span style=\"color: var(--" + hitColor +
                    ")\">" + std::to_string(hitCount) + "</span></td>\n";
            html += "            </tr>\n";
          }

          html += "          </tbody>\n";
          html += "        </table>\n";
          html += "      </div>\n";
        }
      }
      // Close coverpoints collapse-content
      html += "      </div>\n";
    }

    // Cross coverage section
    auto crossIt = crossCoverageData.find(cg);
    if (crossIt != crossCoverageData.end() && !crossIt->second.crosses.empty()) {
      html += "      <h3 class=\"collapsible\" onclick=\"toggleCollapse(this)\">Cross Coverage</h3>\n";
      html += "      <div class=\"collapse-content\">\n";
      html += "      <table class=\"sortable-table\">\n";
      html += "        <thead>\n";
      html += "          <tr>\n";
      html += "            <th>Cross Name</th>\n";
      html += "            <th>Coverpoints</th>\n";
      html += "            <th>Bins Hit</th>\n";
      html += "            <th>Coverage</th>\n";
      html += "          </tr>\n";
      html += "        </thead>\n";
      html += "        <tbody>\n";

      for (size_t crossIdx = 0; crossIdx < crossIt->second.crosses.size(); ++crossIdx) {
        const auto &cross = crossIt->second.crosses[crossIdx];
        double crossCov = __moore_cross_get_coverage(cg, static_cast<int32_t>(crossIdx));
        int64_t binsHit = __moore_cross_get_bins_hit(cg, static_cast<int32_t>(crossIdx));

        html += "          <tr>\n";
        html += "            <td>" + std::string(cross.name ? cross.name : "(unnamed)") + "</td>\n";

        // List crossed coverpoints
        std::string cpList;
        for (int32_t cpIdx = 0; cpIdx < cross.num_cps; ++cpIdx) {
          if (cpIdx > 0) cpList += ", ";
          int32_t realCpIdx = cross.cp_indices[cpIdx];
          if (realCpIdx >= 0 && realCpIdx < cg->num_coverpoints &&
              cg->coverpoints[realCpIdx] && cg->coverpoints[realCpIdx]->name) {
            cpList += cg->coverpoints[realCpIdx]->name;
          } else {
            cpList += "cp" + std::to_string(realCpIdx);
          }
        }
        html += "            <td>" + cpList + "</td>\n";
        html += "            <td>" + std::to_string(binsHit) + "</td>\n";

        // Color coding: green (100%), yellow (50-99%), red (<50%)
        std::snprintf(buf, sizeof(buf), "%.1f%%", crossCov);
        std::string crossColorVar = crossCov >= 100.0 ? "success" :
                                    (crossCov >= 50.0 ? "warning" : "danger");
        html += "            <td><span style=\"color: var(--" + crossColorVar +
                ")\">" + std::string(buf) + "</span></td>\n";
        html += "          </tr>\n";
      }

      html += "        </tbody>\n";
      html += "      </table>\n";
      // Close cross coverage collapse-content
      html += "      </div>\n";
    }

    // Close covergroup collapse-content
    html += "      </div>\n";
    html += "    </div>\n";
  }

  html += "  </div>\n";

  // JavaScript for interactivity
  html += "  <script>\n";
  html += "    function toggleCollapse(element) {\n";
  html += "      element.classList.toggle('collapsed');\n";
  html += "      const content = element.nextElementSibling;\n";
  html += "      if (content && content.classList.contains('collapse-content')) {\n";
  html += "        content.classList.toggle('collapsed');\n";
  html += "      }\n";
  html += "    }\n";
  html += "\n";
  html += "    function expandAll() {\n";
  html += "      document.querySelectorAll('.collapsible').forEach(el => el.classList.remove('collapsed'));\n";
  html += "      document.querySelectorAll('.collapse-content').forEach(el => el.classList.remove('collapsed'));\n";
  html += "    }\n";
  html += "\n";
  html += "    function collapseAll() {\n";
  html += "      document.querySelectorAll('.collapsible').forEach(el => el.classList.add('collapsed'));\n";
  html += "      document.querySelectorAll('.collapse-content').forEach(el => el.classList.add('collapsed'));\n";
  html += "    }\n";
  html += "\n";
  html += "    function filterCovergroups() {\n";
  html += "      const search = document.getElementById('searchInput').value.toLowerCase();\n";
  html += "      const status = document.getElementById('statusFilter').value;\n";
  html += "      const coverage = document.getElementById('coverageFilter').value;\n";
  html += "\n";
  html += "      document.querySelectorAll('.covergroup').forEach(cg => {\n";
  html += "        const name = cg.dataset.name.toLowerCase();\n";
  html += "        const cgStatus = cg.dataset.status;\n";
  html += "        const cgCoverage = cg.dataset.coverage;\n";
  html += "\n";
  html += "        const matchesSearch = !search || name.includes(search);\n";
  html += "        const matchesStatus = status === 'all' || cgStatus === status;\n";
  html += "        let matchesCoverage = true;\n";
  html += "        if (coverage === '100') matchesCoverage = cgCoverage === '100';\n";
  html += "        else if (coverage === 'high') matchesCoverage = cgCoverage === 'high' || cgCoverage === '100';\n";
  html += "        else if (coverage === 'medium') matchesCoverage = cgCoverage === 'medium';\n";
  html += "        else if (coverage === 'low') matchesCoverage = cgCoverage === 'low';\n";
  html += "\n";
  html += "        cg.classList.toggle('hidden', !(matchesSearch && matchesStatus && matchesCoverage));\n";
  html += "      });\n";
  html += "    }\n";
  html += "\n";
  html += "    function sortTable(header, columnIndex) {\n";
  html += "      const table = header.closest('table');\n";
  html += "      const tbody = table.querySelector('tbody');\n";
  html += "      const rows = Array.from(tbody.querySelectorAll('tr'));\n";
  html += "      const isAsc = header.dataset.sortDir !== 'asc';\n";
  html += "      header.dataset.sortDir = isAsc ? 'asc' : 'desc';\n";
  html += "\n";
  html += "      rows.sort((a, b) => {\n";
  html += "        const aVal = a.cells[columnIndex].textContent.trim();\n";
  html += "        const bVal = b.cells[columnIndex].textContent.trim();\n";
  html += "        const aNum = parseFloat(aVal.replace(/[^\\d.-]/g, ''));\n";
  html += "        const bNum = parseFloat(bVal.replace(/[^\\d.-]/g, ''));\n";
  html += "        if (!isNaN(aNum) && !isNaN(bNum)) {\n";
  html += "          return isAsc ? aNum - bNum : bNum - aNum;\n";
  html += "        }\n";
  html += "        return isAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);\n";
  html += "      });\n";
  html += "\n";
  html += "      rows.forEach(row => tbody.appendChild(row));\n";
  html += "    }\n";
  html += "  </script>\n";
  html += "</body>\n";
  html += "</html>\n";

  std::fwrite(html.c_str(), 1, html.size(), fp);
  std::fclose(fp);

  return 0;
}

//===----------------------------------------------------------------------===//
// Text Coverage Report
//===----------------------------------------------------------------------===//
//
// Text-based coverage report functions for CI/automation use.
// These provide simple, parseable output that is faster to generate than HTML.
//

namespace {

/// Generate text coverage report as a string.
/// @param verbosity 0=summary only, 1=normal (covergroups+coverpoints), 2=detailed (includes bins)
std::string generateCoverageText(int32_t verbosity) {
  std::string report;
  char buf[256];

  // Get overall statistics
  double totalCoverage = __moore_coverage_get_total();
  int32_t numCovergroups = __moore_coverage_get_num_covergroups();
  int32_t totalCoverpoints = 0;
  int64_t totalHits = 0;
  int32_t holesCount = 0;  // Bins with 0 hits

  for (auto *cg : registeredCovergroups) {
    if (cg) {
      totalCoverpoints += cg->num_coverpoints;
      for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
        if (cg->coverpoints[i]) {
          totalHits += cg->coverpoints[i]->hits;
          // Count bins with 0 hits
          auto binIt = explicitBinData.find(cg->coverpoints[i]);
          if (binIt != explicitBinData.end()) {
            for (const auto &bin : binIt->second.bins) {
              if (bin.kind == MOORE_BIN_KIND_NORMAL && bin.hit_count == 0)
                ++holesCount;
            }
          }
        }
      }
    }
  }

  // Header
  report += "Coverage Report\n";
  report += "================================\n";
  std::snprintf(buf, sizeof(buf), "Overall Coverage: %.1f%%\n", totalCoverage);
  report += buf;
  report += "\n";

  // Summary statistics
  report += "Summary:\n";
  std::snprintf(buf, sizeof(buf), "  Covergroups: %d\n", numCovergroups);
  report += buf;
  std::snprintf(buf, sizeof(buf), "  Coverpoints: %d\n", totalCoverpoints);
  report += buf;
  std::snprintf(buf, sizeof(buf), "  Total Samples: %ld\n", static_cast<long>(totalHits));
  report += buf;
  if (holesCount > 0) {
    std::snprintf(buf, sizeof(buf), "  Coverage Holes: %d\n", holesCount);
    report += buf;
  }
  report += "\n";

  // If summary only, return here
  if (verbosity == MOORE_TEXT_REPORT_SUMMARY) {
    return report;
  }

  // Per-covergroup details
  for (auto *cg : registeredCovergroups) {
    if (!cg)
      continue;

    double cgCoverage = __moore_covergroup_get_coverage(cg);
    double goal = __moore_covergroup_get_goal(cg);
    bool goalMet = cgCoverage >= goal;

    report += "Covergroup: ";
    report += cg->name ? cg->name : "(unnamed)";
    report += "\n";

    std::snprintf(buf, sizeof(buf), "  Coverage: %.1f%% (goal: %.0f%%)%s\n",
                  cgCoverage, goal, goalMet ? "" : " <-- NOT MET");
    report += buf;

    // Coverpoints
    for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
      auto *cp = cg->coverpoints[i];
      if (!cp)
        continue;

      double cpCoverage = __moore_coverpoint_get_coverage(cg, i);
      std::snprintf(buf, sizeof(buf), "  Coverpoint: %s (%.1f%%)\n",
                    cp->name ? cp->name : "(unnamed)", cpCoverage);
      report += buf;

      // Show bins in detailed mode
      if (verbosity == MOORE_TEXT_REPORT_DETAILED) {
        auto binIt = explicitBinData.find(cp);
        if (binIt != explicitBinData.end() && !binIt->second.bins.empty()) {
          for (size_t binIdx = 0; binIdx < binIt->second.bins.size(); ++binIdx) {
            const auto &bin = binIt->second.bins[binIdx];

            // Skip ignore/illegal bins in summary
            if (bin.kind != MOORE_BIN_KIND_NORMAL)
              continue;

            // Get hit count
            int64_t hitCount = (cp->bins && static_cast<int32_t>(binIdx) < cp->num_bins)
                                   ? cp->bins[binIdx]
                                   : bin.hit_count;

            // Format bin info
            std::string binRange;
            if (bin.type == MOORE_BIN_VALUE) {
              std::snprintf(buf, sizeof(buf), "%ld", static_cast<long>(bin.low));
              binRange = buf;
            } else if (bin.type == MOORE_BIN_RANGE) {
              std::snprintf(buf, sizeof(buf), "%ld..%ld",
                            static_cast<long>(bin.low), static_cast<long>(bin.high));
              binRange = buf;
            } else if (bin.type == MOORE_BIN_WILDCARD) {
              binRange = "wildcard";
            } else if (bin.type == MOORE_BIN_TRANSITION) {
              binRange = "transition";
            }

            std::snprintf(buf, sizeof(buf), "    bin %s: %ld hits",
                          bin.name ? bin.name : "(unnamed)",
                          static_cast<long>(hitCount));
            report += buf;

            if (!binRange.empty()) {
              report += " [";
              report += binRange;
              report += "]";
            }

            if (hitCount == 0) {
              report += "  <-- HOLE";
            }
            report += "\n";
          }
        }
      }
    }

    // Cross coverage
    auto crossIt = crossCoverageData.find(cg);
    if (crossIt != crossCoverageData.end() && !crossIt->second.crosses.empty()) {
      for (size_t crossIdx = 0; crossIdx < crossIt->second.crosses.size(); ++crossIdx) {
        const auto &cross = crossIt->second.crosses[crossIdx];
        double crossCov = __moore_cross_get_coverage(cg, static_cast<int32_t>(crossIdx));
        int64_t binsHit = __moore_cross_get_bins_hit(cg, static_cast<int32_t>(crossIdx));

        // Build coverpoint list
        std::string cpList;
        for (int32_t cpIdx = 0; cpIdx < cross.num_cps; ++cpIdx) {
          if (cpIdx > 0)
            cpList += " x ";
          int32_t realCpIdx = cross.cp_indices[cpIdx];
          if (realCpIdx >= 0 && realCpIdx < cg->num_coverpoints &&
              cg->coverpoints[realCpIdx] && cg->coverpoints[realCpIdx]->name) {
            cpList += cg->coverpoints[realCpIdx]->name;
          } else {
            std::snprintf(buf, sizeof(buf), "cp%d", realCpIdx);
            cpList += buf;
          }
        }

        std::snprintf(buf, sizeof(buf), "  Cross: %s (%.1f%%, %ld bins hit)\n",
                      cross.name ? cross.name : "(unnamed)", crossCov,
                      static_cast<long>(binsHit));
        report += buf;

        if (verbosity == MOORE_TEXT_REPORT_DETAILED) {
          report += "    Coverpoints: ";
          report += cpList;
          report += "\n";
        }
      }
    }

    report += "\n";
  }

  // Coverage holes summary
  if (holesCount > 0 && verbosity >= MOORE_TEXT_REPORT_NORMAL) {
    report += "================================\n";
    report += "Coverage Holes (0 hits):\n";
    report += "================================\n";

    for (auto *cg : registeredCovergroups) {
      if (!cg)
        continue;

      for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
        auto *cp = cg->coverpoints[i];
        if (!cp)
          continue;

        auto binIt = explicitBinData.find(cp);
        if (binIt != explicitBinData.end()) {
          for (size_t binIdx = 0; binIdx < binIt->second.bins.size(); ++binIdx) {
            const auto &bin = binIt->second.bins[binIdx];
            if (bin.kind != MOORE_BIN_KIND_NORMAL)
              continue;

            int64_t hitCount = (cp->bins && static_cast<int32_t>(binIdx) < cp->num_bins)
                                   ? cp->bins[binIdx]
                                   : bin.hit_count;

            if (hitCount == 0) {
              std::snprintf(buf, sizeof(buf), "  %s.%s.%s\n",
                            cg->name ? cg->name : "(unnamed)",
                            cp->name ? cp->name : "(unnamed)",
                            bin.name ? bin.name : "(unnamed)");
              report += buf;
            }
          }
        }
      }
    }
    report += "\n";
  }

  return report;
}

} // anonymous namespace

extern "C" int32_t __moore_coverage_report_text(const char *filename,
                                                  int32_t verbosity) {
  if (!filename)
    return 1;

  FILE *fp = std::fopen(filename, "w");
  if (!fp)
    return 1;

  std::string report = generateCoverageText(verbosity);
  std::fwrite(report.c_str(), 1, report.size(), fp);
  std::fclose(fp);

  return 0;
}

extern "C" char *__moore_coverage_report_summary(void) {
  // Generate a brief one-line summary
  double totalCoverage = __moore_coverage_get_total();
  int32_t numCovergroups = __moore_coverage_get_num_covergroups();

  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "Coverage: %.1f%% (%d covergroups)",
                totalCoverage, numCovergroups);

  char *result = static_cast<char *>(std::malloc(std::strlen(buf) + 1));
  if (!result)
    return nullptr;

  std::strcpy(result, buf);
  return result;
}

extern "C" void __moore_coverage_print_summary(void) {
  double totalCoverage = __moore_coverage_get_total();
  int32_t numCovergroups = __moore_coverage_get_num_covergroups();
  int32_t totalCoverpoints = 0;
  int64_t totalHits = 0;

  for (auto *cg : registeredCovergroups) {
    if (cg) {
      totalCoverpoints += cg->num_coverpoints;
      for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
        if (cg->coverpoints[i])
          totalHits += cg->coverpoints[i]->hits;
      }
    }
  }

  std::printf("\n");
  std::printf("================================\n");
  std::printf("Coverage Summary\n");
  std::printf("================================\n");
  std::printf("Overall Coverage: %.1f%%\n", totalCoverage);
  std::printf("Covergroups:      %d\n", numCovergroups);
  std::printf("Coverpoints:      %d\n", totalCoverpoints);
  std::printf("Total Samples:    %ld\n", static_cast<long>(totalHits));
  std::printf("================================\n");
  std::printf("\n");
}

extern "C" char *__moore_coverage_get_text_report(int32_t verbosity) {
  std::string report = generateCoverageText(verbosity);

  char *result = static_cast<char *>(std::malloc(report.size() + 1));
  if (!result)
    return nullptr;

  std::memcpy(result, report.c_str(), report.size() + 1);
  return result;
}

extern "C" void __moore_coverage_print_text(int32_t verbosity) {
  std::string report = generateCoverageText(verbosity);
  std::printf("%s", report.c_str());
}

extern "C" void __moore_coverage_report_on_finish(int32_t verbosity) {
  // This function is intended to be called at $finish to print coverage
  // Auto-detect verbosity if -1: use normal for few covergroups, summary for many
  int32_t effectiveVerbosity = verbosity;
  if (effectiveVerbosity < 0) {
    int32_t numCg = __moore_coverage_get_num_covergroups();
    effectiveVerbosity = (numCg > 10) ? MOORE_TEXT_REPORT_SUMMARY : MOORE_TEXT_REPORT_NORMAL;
  }

  std::printf("\n");
  std::printf("================================================================================\n");
  std::printf("                          CIRCT Coverage Report at Finish\n");
  std::printf("================================================================================\n");

  std::string report = generateCoverageText(effectiveVerbosity);
  std::printf("%s", report.c_str());

  // Add pass/fail summary
  double totalCoverage = __moore_coverage_get_total();
  int32_t passedGoals = 0;
  int32_t totalGoals = 0;

  for (auto *cg : registeredCovergroups) {
    if (!cg)
      continue;
    ++totalGoals;
    double cgCoverage = __moore_covergroup_get_coverage(cg);
    double goal = __moore_covergroup_get_goal(cg);
    if (cgCoverage >= goal)
      ++passedGoals;
  }

  std::printf("================================================================================\n");
  if (totalGoals > 0) {
    std::printf("Goals: %d/%d covergroups passed (%.1f%%)\n",
                passedGoals, totalGoals, (100.0 * passedGoals) / totalGoals);
    if (passedGoals == totalGoals && totalCoverage >= 100.0) {
      std::printf("Status: PASS - All coverage goals met!\n");
    } else if (passedGoals == totalGoals) {
      std::printf("Status: PASS - All covergroup goals met (overall: %.1f%%)\n", totalCoverage);
    } else {
      std::printf("Status: FAIL - %d covergroup(s) did not meet goal\n", totalGoals - passedGoals);
    }
  } else {
    std::printf("No covergroups registered.\n");
  }
  std::printf("================================================================================\n\n");
}

//===----------------------------------------------------------------------===//
// Coverage Database Save/Load/Merge Operations
//===----------------------------------------------------------------------===//
//
// These functions support coverage database persistence and merging.
// This enables verification flows that combine coverage from multiple
// simulation runs: run1.db + run2.db + run3.db -> merged.db
//

/// Internal structure to hold loaded coverage database data.
/// This must be outside the anonymous namespace to match the forward
/// declaration in MooreRuntime.h (MooreCoverageDBHandle).
struct MooreCoverageDB {
  /// Covergroup data from the loaded database.
  struct CoverpointData {
    std::string name;
    int64_t hits;
    int64_t minVal;
    int64_t maxVal;
    std::map<int64_t, int64_t> valueCounts; // value -> hit count
    std::vector<MooreCoverageBin> bins;
    double coverage;
  };

  struct CovergroupData {
    std::string name;
    double coverage;
    std::vector<CoverpointData> coverpoints;
  };

  /// Internal metadata storage (strings are owned by this struct).
  struct MetadataStorage {
    std::string testName;
    int64_t timestamp = 0;
    std::string simulator;
    std::string version;
    std::string comment;
  };

  std::vector<CovergroupData> covergroups;
  MetadataStorage metadataStorage;
  MooreCoverageMetadata metadata = {nullptr, 0, nullptr, nullptr, nullptr};
  bool hasMetadata = false;

  /// Update the metadata pointers to point to the storage strings.
  void updateMetadataPointers() {
    metadata.test_name = metadataStorage.testName.empty()
                             ? nullptr
                             : metadataStorage.testName.c_str();
    metadata.timestamp = metadataStorage.timestamp;
    metadata.simulator = metadataStorage.simulator.empty()
                             ? nullptr
                             : metadataStorage.simulator.c_str();
    metadata.version = metadataStorage.version.empty()
                           ? nullptr
                           : metadataStorage.version.c_str();
    metadata.comment = metadataStorage.comment.empty()
                           ? nullptr
                           : metadataStorage.comment.c_str();
  }

  /// Calculate total coverage across all covergroups.
  double getTotalCoverage() const {
    if (covergroups.empty())
      return 0.0;
    double total = 0.0;
    for (const auto &cg : covergroups)
      total += cg.coverage;
    return total / covergroups.size();
  }
};

namespace {

/// Simple JSON parser for coverage database files.
/// This is a minimal parser that handles the specific JSON format we generate.
class SimpleJsonParser {
public:
  explicit SimpleJsonParser(const std::string &json) : json(json), pos(0) {}

  bool parse(MooreCoverageDB &db) {
    skipWhitespace();
    if (!expect('{'))
      return false;

    while (pos < json.size()) {
      skipWhitespace();
      if (peek() == '}')
        break;

      std::string key = parseString();
      skipWhitespace();
      if (!expect(':'))
        return false;
      skipWhitespace();

      if (key == "coverage_report") {
        if (!parseCoverageReport(db))
          return false;
      } else {
        // Skip unknown keys
        if (!skipValue())
          return false;
      }

      skipWhitespace();
      if (peek() == ',')
        advance();
    }

    expect('}');
    return true;
  }

private:
  const std::string &json;
  size_t pos;

  char peek() const { return pos < json.size() ? json[pos] : '\0'; }
  void advance() { if (pos < json.size()) ++pos; }

  void skipWhitespace() {
    while (pos < json.size() && std::isspace(json[pos]))
      ++pos;
  }

  bool expect(char c) {
    skipWhitespace();
    if (peek() != c)
      return false;
    advance();
    return true;
  }

  std::string parseString() {
    skipWhitespace();
    if (peek() != '"')
      return "";
    advance();
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
      if (json[pos] == '\\' && pos + 1 < json.size()) {
        advance();
        switch (json[pos]) {
        case 'n': result += '\n'; break;
        case 'r': result += '\r'; break;
        case 't': result += '\t'; break;
        case '"': result += '"'; break;
        case '\\': result += '\\'; break;
        default: result += json[pos]; break;
        }
      } else {
        result += json[pos];
      }
      advance();
    }
    advance(); // Skip closing quote
    return result;
  }

  int64_t parseNumber() {
    skipWhitespace();
    bool negative = false;
    if (peek() == '-') {
      negative = true;
      advance();
    }
    int64_t result = 0;
    while (pos < json.size() && std::isdigit(json[pos])) {
      result = result * 10 + (json[pos] - '0');
      advance();
    }
    // Skip decimal part if present
    if (peek() == '.') {
      advance();
      while (pos < json.size() && std::isdigit(json[pos]))
        advance();
    }
    return negative ? -result : result;
  }

  double parseDouble() {
    skipWhitespace();
    size_t startPos = pos;
    if (peek() == '-')
      advance();
    while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '.'))
      advance();
    std::string numStr = json.substr(startPos, pos - startPos);
    return std::stod(numStr);
  }

  bool skipValue() {
    skipWhitespace();
    char c = peek();
    if (c == '"') {
      parseString();
      return true;
    } else if (c == '[') {
      advance();
      int depth = 1;
      while (depth > 0 && pos < json.size()) {
        if (json[pos] == '[') ++depth;
        else if (json[pos] == ']') --depth;
        else if (json[pos] == '"') {
          advance();
          while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\') advance();
            advance();
          }
        }
        advance();
      }
      return true;
    } else if (c == '{') {
      advance();
      int depth = 1;
      while (depth > 0 && pos < json.size()) {
        if (json[pos] == '{') ++depth;
        else if (json[pos] == '}') --depth;
        else if (json[pos] == '"') {
          advance();
          while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\') advance();
            advance();
          }
        }
        advance();
      }
      return true;
    } else if (c == 'n') { // null
      pos += 4;
      return true;
    } else if (c == 't') { // true
      pos += 4;
      return true;
    } else if (c == 'f') { // false
      pos += 5;
      return true;
    } else {
      // Number
      parseNumber();
      return true;
    }
  }

  bool parseCoverageReport(MooreCoverageDB &db) {
    if (!expect('{'))
      return false;

    while (pos < json.size()) {
      skipWhitespace();
      if (peek() == '}')
        break;

      std::string key = parseString();
      skipWhitespace();
      if (!expect(':'))
        return false;
      skipWhitespace();

      if (key == "covergroups") {
        if (!parseCovergroups(db))
          return false;
      } else {
        if (!skipValue())
          return false;
      }

      skipWhitespace();
      if (peek() == ',')
        advance();
    }

    return expect('}');
  }

  bool parseCovergroups(MooreCoverageDB &db) {
    if (!expect('['))
      return false;

    while (pos < json.size()) {
      skipWhitespace();
      if (peek() == ']')
        break;

      MooreCoverageDB::CovergroupData cg;
      if (!parseCovergroup(cg))
        return false;
      db.covergroups.push_back(std::move(cg));

      skipWhitespace();
      if (peek() == ',')
        advance();
    }

    return expect(']');
  }

  bool parseCovergroup(MooreCoverageDB::CovergroupData &cg) {
    if (!expect('{'))
      return false;

    while (pos < json.size()) {
      skipWhitespace();
      if (peek() == '}')
        break;

      std::string key = parseString();
      skipWhitespace();
      if (!expect(':'))
        return false;
      skipWhitespace();

      if (key == "name") {
        cg.name = parseString();
      } else if (key == "coverage_percent") {
        cg.coverage = parseDouble();
      } else if (key == "coverpoints") {
        if (!parseCoverpoints(cg))
          return false;
      } else {
        if (!skipValue())
          return false;
      }

      skipWhitespace();
      if (peek() == ',')
        advance();
    }

    return expect('}');
  }

  bool parseCoverpoints(MooreCoverageDB::CovergroupData &cg) {
    if (!expect('['))
      return false;

    while (pos < json.size()) {
      skipWhitespace();
      if (peek() == ']')
        break;

      MooreCoverageDB::CoverpointData cp;
      if (!parseCoverpoint(cp))
        return false;
      cg.coverpoints.push_back(std::move(cp));

      skipWhitespace();
      if (peek() == ',')
        advance();
    }

    return expect(']');
  }

  bool parseCoverpoint(MooreCoverageDB::CoverpointData &cp) {
    if (!expect('{'))
      return false;

    while (pos < json.size()) {
      skipWhitespace();
      if (peek() == '}')
        break;

      std::string key = parseString();
      skipWhitespace();
      if (!expect(':'))
        return false;
      skipWhitespace();

      if (key == "name") {
        cp.name = parseString();
      } else if (key == "coverage_percent") {
        cp.coverage = parseDouble();
      } else if (key == "total_hits") {
        cp.hits = parseNumber();
      } else if (key == "min_value") {
        cp.minVal = parseNumber();
      } else if (key == "max_value") {
        cp.maxVal = parseNumber();
      } else if (key == "bins") {
        if (!parseBins(cp))
          return false;
      } else if (key == "top_values") {
        if (!parseTopValues(cp))
          return false;
      } else {
        if (!skipValue())
          return false;
      }

      skipWhitespace();
      if (peek() == ',')
        advance();
    }

    return expect('}');
  }

  bool parseBins(MooreCoverageDB::CoverpointData &cp) {
    if (!expect('['))
      return false;

    while (pos < json.size()) {
      skipWhitespace();
      if (peek() == ']')
        break;

      MooreCoverageBin bin;
      bin.name = nullptr;
      bin.type = MOORE_BIN_VALUE;
      bin.low = 0;
      bin.high = 0;
      bin.hit_count = 0;

      if (!parseBin(bin, cp))
        return false;
      cp.bins.push_back(bin);

      skipWhitespace();
      if (peek() == ',')
        advance();
    }

    return expect(']');
  }

  bool parseBin(MooreCoverageBin &bin, MooreCoverageDB::CoverpointData &cp) {
    if (!expect('{'))
      return false;

    std::string binName;
    while (pos < json.size()) {
      skipWhitespace();
      if (peek() == '}')
        break;

      std::string key = parseString();
      skipWhitespace();
      if (!expect(':'))
        return false;
      skipWhitespace();

      if (key == "name") {
        binName = parseString();
      } else if (key == "type") {
        std::string typeStr = parseString();
        if (typeStr == "value") bin.type = MOORE_BIN_VALUE;
        else if (typeStr == "range") bin.type = MOORE_BIN_RANGE;
        else if (typeStr == "wildcard") bin.type = MOORE_BIN_WILDCARD;
        else if (typeStr == "transition") bin.type = MOORE_BIN_TRANSITION;
      } else if (key == "low") {
        bin.low = parseNumber();
      } else if (key == "high") {
        bin.high = parseNumber();
      } else if (key == "hit_count") {
        bin.hit_count = parseNumber();
      } else {
        if (!skipValue())
          return false;
      }

      skipWhitespace();
      if (peek() == ',')
        advance();
    }

    // Store the bin name (we need to allocate storage for it)
    if (!binName.empty()) {
      char *nameCopy = static_cast<char *>(std::malloc(binName.size() + 1));
      if (nameCopy) {
        std::strcpy(nameCopy, binName.c_str());
        bin.name = nameCopy;
      }
    }

    return expect('}');
  }

  bool parseTopValues(MooreCoverageDB::CoverpointData &cp) {
    if (!expect('['))
      return false;

    while (pos < json.size()) {
      skipWhitespace();
      if (peek() == ']')
        break;

      if (!expect('{'))
        return false;

      int64_t value = 0;
      int64_t count = 0;

      while (pos < json.size()) {
        skipWhitespace();
        if (peek() == '}')
          break;

        std::string key = parseString();
        skipWhitespace();
        if (!expect(':'))
          return false;
        skipWhitespace();

        if (key == "value") {
          value = parseNumber();
        } else if (key == "count") {
          count = parseNumber();
        }

        skipWhitespace();
        if (peek() == ',')
          advance();
      }

      if (!expect('}'))
        return false;

      cp.valueCounts[value] = count;

      skipWhitespace();
      if (peek() == ',')
        advance();
    }

    return expect(']');
  }
};

/// Generate a complete coverage JSON with full value counts (not just top 10).
/// This is used for save/merge operations where we need all the data.
std::string generateFullCoverageJson() {
  std::string json = "{\n";
  json += "  \"coverage_report\": {\n";
  json += "    \"version\": \"1.0\",\n";
  json += "    \"generator\": \"circt-moore-runtime\",\n";
  json += "    \"format\": \"coverage_database\",\n";
  json += "    \"covergroups\": [\n";

  bool firstCg = true;
  for (auto *cg : registeredCovergroups) {
    if (!cg)
      continue;

    if (!firstCg)
      json += ",\n";
    firstCg = false;

    double cgCoverage = __moore_covergroup_get_coverage(cg);

    json += "      {\n";
    json += "        \"name\": " + jsonEscapeString(cg->name) + ",\n";
    json += "        \"coverage_percent\": " + std::to_string(cgCoverage) + ",\n";
    json += "        \"num_coverpoints\": " + std::to_string(cg->num_coverpoints) + ",\n";
    json += "        \"coverpoints\": [\n";

    bool firstCp = true;
    for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
      auto *cp = cg->coverpoints[i];
      if (!cp)
        continue;

      if (!firstCp)
        json += ",\n";
      firstCp = false;

      double cpCoverage = __moore_coverpoint_get_coverage(cg, i);
      auto trackerIt = coverpointTrackers.find(cp);

      json += "          {\n";
      json += "            \"name\": " + jsonEscapeString(cp->name) + ",\n";
      json += "            \"coverage_percent\": " + std::to_string(cpCoverage) + ",\n";
      json += "            \"total_hits\": " + std::to_string(cp->hits) + ",\n";
      json += "            \"min_value\": " + std::to_string(cp->min_val) + ",\n";
      json += "            \"max_value\": " + std::to_string(cp->max_val) + ",\n";

      // Add explicit bins if present
      json += "            \"bins\": [\n";
      auto binIt = explicitBinData.find(cp);
      if (binIt != explicitBinData.end()) {
        bool firstBin = true;
        for (const auto &bin : binIt->second.bins) {
          if (!firstBin)
            json += ",\n";
          firstBin = false;

          const char *binTypeName = "unknown";
          switch (bin.type) {
          case MOORE_BIN_VALUE: binTypeName = "value"; break;
          case MOORE_BIN_RANGE: binTypeName = "range"; break;
          case MOORE_BIN_WILDCARD: binTypeName = "wildcard"; break;
          case MOORE_BIN_TRANSITION: binTypeName = "transition"; break;
          }

          json += "              {\n";
          json += "                \"name\": " + jsonEscapeString(bin.name) + ",\n";
          json += "                \"type\": \"" + std::string(binTypeName) + "\",\n";
          json += "                \"low\": " + std::to_string(bin.low) + ",\n";
          json += "                \"high\": " + std::to_string(bin.high) + ",\n";
          json += "                \"hit_count\": " + std::to_string(bin.hit_count) + "\n";
          json += "              }";
        }
      }
      json += "\n            ],\n";

      // Add ALL value counts (not just top 10) for accurate merging
      json += "            \"top_values\": [\n";
      if (trackerIt != coverpointTrackers.end()) {
        bool firstVal = true;
        for (const auto &kv : trackerIt->second.valueCounts) {
          if (!firstVal)
            json += ",\n";
          firstVal = false;
          json += "              {\"value\": " + std::to_string(kv.first) +
                  ", \"count\": " + std::to_string(kv.second) + "}";
        }
      }
      json += "\n            ]\n";
      json += "          }";
    }

    json += "\n        ]\n";
    json += "      }";
  }

  json += "\n    ]\n";
  json += "  }\n";
  json += "}\n";

  return json;
}

} // anonymous namespace

extern "C" int32_t __moore_coverage_save(const char *filename) {
  if (!filename)
    return 1;

  FILE *fp = std::fopen(filename, "w");
  if (!fp)
    return 1;

  std::string json = generateFullCoverageJson();
  std::fwrite(json.c_str(), 1, json.size(), fp);
  std::fclose(fp);

  return 0;
}

extern "C" MooreCoverageDBHandle __moore_coverage_load(const char *filename) {
  if (!filename)
    return nullptr;

  FILE *fp = std::fopen(filename, "r");
  if (!fp)
    return nullptr;

  // Read the entire file
  std::fseek(fp, 0, SEEK_END);
  long fileSize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  if (fileSize <= 0) {
    std::fclose(fp);
    return nullptr;
  }

  std::string json(fileSize, '\0');
  size_t bytesRead = std::fread(&json[0], 1, fileSize, fp);
  std::fclose(fp);

  if (bytesRead != static_cast<size_t>(fileSize))
    return nullptr;

  // Parse the JSON
  auto *db = new MooreCoverageDB();
  SimpleJsonParser parser(json);
  if (!parser.parse(*db)) {
    delete db;
    return nullptr;
  }

  return db;
}

extern "C" void __moore_coverage_db_free(MooreCoverageDBHandle db) {
  if (!db)
    return;

  // Free any allocated bin names
  for (auto &cg : db->covergroups) {
    for (auto &cp : cg.coverpoints) {
      for (auto &bin : cp.bins) {
        if (bin.name) {
          std::free(const_cast<char *>(bin.name));
        }
      }
    }
  }

  delete db;
}

extern "C" int32_t __moore_coverage_merge(MooreCoverageDBHandle db) {
  if (!db)
    return 1;

  // Merge each covergroup from the loaded database into the current state
  for (const auto &loadedCg : db->covergroups) {
    // Find matching covergroup by name
    MooreCovergroup *targetCg = nullptr;
    for (auto *cg : registeredCovergroups) {
      if (cg && cg->name && loadedCg.name == cg->name) {
        targetCg = cg;
        break;
      }
    }

    if (!targetCg)
      continue; // No matching covergroup found

    // Merge each coverpoint
    for (const auto &loadedCp : loadedCg.coverpoints) {
      // Find matching coverpoint by name
      MooreCoverpoint *targetCp = nullptr;
      for (int32_t i = 0; i < targetCg->num_coverpoints; ++i) {
        auto *cp = targetCg->coverpoints[i];
        if (cp && cp->name && loadedCp.name == cp->name) {
          targetCp = cp;
          break;
        }
      }

      if (!targetCp)
        continue; // No matching coverpoint found

      // Merge hit counts
      targetCp->hits += loadedCp.hits;

      // Merge min/max values
      if (loadedCp.minVal < targetCp->min_val)
        targetCp->min_val = loadedCp.minVal;
      if (loadedCp.maxVal > targetCp->max_val)
        targetCp->max_val = loadedCp.maxVal;

      // Merge value counts
      auto trackerIt = coverpointTrackers.find(targetCp);
      if (trackerIt != coverpointTrackers.end()) {
        for (const auto &kv : loadedCp.valueCounts) {
          trackerIt->second.valueCounts[kv.first] += kv.second;
        }
      }

      // Merge explicit bin hit counts
      auto binIt = explicitBinData.find(targetCp);
      if (binIt != explicitBinData.end() && !loadedCp.bins.empty()) {
        // Match bins by name and merge hit counts
        for (const auto &loadedBin : loadedCp.bins) {
          for (auto &targetBin : binIt->second.bins) {
            // Match by name if available, otherwise by range
            bool match = false;
            if (loadedBin.name && targetBin.name) {
              match = std::strcmp(loadedBin.name, targetBin.name) == 0;
            } else {
              match = (loadedBin.low == targetBin.low &&
                       loadedBin.high == targetBin.high &&
                       loadedBin.type == targetBin.type);
            }
            if (match) {
              targetBin.hit_count += loadedBin.hit_count;
              break;
            }
          }
        }

        // Also update the legacy bins array if present
        if (targetCp->bins && targetCp->num_bins > 0) {
          for (int32_t i = 0; i < targetCp->num_bins &&
               i < static_cast<int32_t>(binIt->second.bins.size()); ++i) {
            targetCp->bins[i] = binIt->second.bins[i].hit_count;
          }
        }
      }
    }
  }

  return 0;
}

extern "C" int32_t __moore_coverage_merge_file(const char *filename) {
  MooreCoverageDBHandle db = __moore_coverage_load(filename);
  if (!db)
    return 1;

  int32_t result = __moore_coverage_merge(db);
  __moore_coverage_db_free(db);
  return result;
}

extern "C" int32_t __moore_coverage_merge_files(const char *file1,
                                                 const char *file2,
                                                 const char *output) {
  if (!file1 || !file2 || !output)
    return 1;

  // Load both databases
  MooreCoverageDBHandle db1 = __moore_coverage_load(file1);
  if (!db1)
    return 1;

  MooreCoverageDBHandle db2 = __moore_coverage_load(file2);
  if (!db2) {
    __moore_coverage_db_free(db1);
    return 1;
  }

  // Create a merged database by combining data from both
  MooreCoverageDB mergedDb;

  // Start with data from db1
  for (const auto &cg1 : db1->covergroups) {
    MooreCoverageDB::CovergroupData mergedCg;
    mergedCg.name = cg1.name;
    mergedCg.coverage = cg1.coverage; // Will be recalculated

    for (const auto &cp1 : cg1.coverpoints) {
      MooreCoverageDB::CoverpointData mergedCp;
      mergedCp.name = cp1.name;
      mergedCp.hits = cp1.hits;
      mergedCp.minVal = cp1.minVal;
      mergedCp.maxVal = cp1.maxVal;
      mergedCp.coverage = cp1.coverage;
      mergedCp.valueCounts = cp1.valueCounts;

      // Copy bins
      for (const auto &bin : cp1.bins) {
        MooreCoverageBin binCopy;
        binCopy.type = bin.type;
        binCopy.low = bin.low;
        binCopy.high = bin.high;
        binCopy.hit_count = bin.hit_count;
        if (bin.name) {
          char *nameCopy = static_cast<char *>(std::malloc(std::strlen(bin.name) + 1));
          std::strcpy(nameCopy, bin.name);
          binCopy.name = nameCopy;
        } else {
          binCopy.name = nullptr;
        }
        mergedCp.bins.push_back(binCopy);
      }

      mergedCg.coverpoints.push_back(std::move(mergedCp));
    }

    mergedDb.covergroups.push_back(std::move(mergedCg));
  }

  // Merge data from db2
  for (const auto &cg2 : db2->covergroups) {
    // Find matching covergroup
    MooreCoverageDB::CovergroupData *targetCg = nullptr;
    for (auto &cg : mergedDb.covergroups) {
      if (cg.name == cg2.name) {
        targetCg = &cg;
        break;
      }
    }

    if (!targetCg) {
      // No matching covergroup, add it as new
      MooreCoverageDB::CovergroupData newCg;
      newCg.name = cg2.name;
      newCg.coverage = cg2.coverage;

      for (const auto &cp2 : cg2.coverpoints) {
        MooreCoverageDB::CoverpointData newCp;
        newCp.name = cp2.name;
        newCp.hits = cp2.hits;
        newCp.minVal = cp2.minVal;
        newCp.maxVal = cp2.maxVal;
        newCp.coverage = cp2.coverage;
        newCp.valueCounts = cp2.valueCounts;

        for (const auto &bin : cp2.bins) {
          MooreCoverageBin binCopy;
          binCopy.type = bin.type;
          binCopy.low = bin.low;
          binCopy.high = bin.high;
          binCopy.hit_count = bin.hit_count;
          if (bin.name) {
            char *nameCopy = static_cast<char *>(std::malloc(std::strlen(bin.name) + 1));
            std::strcpy(nameCopy, bin.name);
            binCopy.name = nameCopy;
          } else {
            binCopy.name = nullptr;
          }
          newCp.bins.push_back(binCopy);
        }

        newCg.coverpoints.push_back(std::move(newCp));
      }

      mergedDb.covergroups.push_back(std::move(newCg));
      continue;
    }

    // Merge coverpoints
    for (const auto &cp2 : cg2.coverpoints) {
      MooreCoverageDB::CoverpointData *targetCp = nullptr;
      for (auto &cp : targetCg->coverpoints) {
        if (cp.name == cp2.name) {
          targetCp = &cp;
          break;
        }
      }

      if (!targetCp) {
        // No matching coverpoint, add as new
        MooreCoverageDB::CoverpointData newCp;
        newCp.name = cp2.name;
        newCp.hits = cp2.hits;
        newCp.minVal = cp2.minVal;
        newCp.maxVal = cp2.maxVal;
        newCp.coverage = cp2.coverage;
        newCp.valueCounts = cp2.valueCounts;

        for (const auto &bin : cp2.bins) {
          MooreCoverageBin binCopy;
          binCopy.type = bin.type;
          binCopy.low = bin.low;
          binCopy.high = bin.high;
          binCopy.hit_count = bin.hit_count;
          if (bin.name) {
            char *nameCopy = static_cast<char *>(std::malloc(std::strlen(bin.name) + 1));
            std::strcpy(nameCopy, bin.name);
            binCopy.name = nameCopy;
          } else {
            binCopy.name = nullptr;
          }
          newCp.bins.push_back(binCopy);
        }

        targetCg->coverpoints.push_back(std::move(newCp));
        continue;
      }

      // Merge the coverpoint data
      targetCp->hits += cp2.hits;
      if (cp2.minVal < targetCp->minVal)
        targetCp->minVal = cp2.minVal;
      if (cp2.maxVal > targetCp->maxVal)
        targetCp->maxVal = cp2.maxVal;

      // Merge value counts
      for (const auto &kv : cp2.valueCounts) {
        targetCp->valueCounts[kv.first] += kv.second;
      }

      // Merge bin hit counts
      for (const auto &bin2 : cp2.bins) {
        bool found = false;
        for (auto &targetBin : targetCp->bins) {
          bool match = false;
          if (bin2.name && targetBin.name) {
            match = std::strcmp(bin2.name, targetBin.name) == 0;
          } else {
            match = (bin2.low == targetBin.low &&
                     bin2.high == targetBin.high &&
                     bin2.type == targetBin.type);
          }
          if (match) {
            targetBin.hit_count += bin2.hit_count;
            found = true;
            break;
          }
        }
        if (!found) {
          // Add new bin
          MooreCoverageBin binCopy;
          binCopy.type = bin2.type;
          binCopy.low = bin2.low;
          binCopy.high = bin2.high;
          binCopy.hit_count = bin2.hit_count;
          if (bin2.name) {
            char *nameCopy = static_cast<char *>(std::malloc(std::strlen(bin2.name) + 1));
            std::strcpy(nameCopy, bin2.name);
            binCopy.name = nameCopy;
          } else {
            binCopy.name = nullptr;
          }
          targetCp->bins.push_back(binCopy);
        }
      }
    }
  }

  // Write the merged database to the output file
  FILE *fp = std::fopen(output, "w");
  if (!fp) {
    __moore_coverage_db_free(db1);
    __moore_coverage_db_free(db2);
    return 1;
  }

  // Generate JSON for the merged database
  std::string json = "{\n";
  json += "  \"coverage_report\": {\n";
  json += "    \"version\": \"1.0\",\n";
  json += "    \"generator\": \"circt-moore-runtime\",\n";
  json += "    \"format\": \"coverage_database\",\n";
  json += "    \"merged_from\": [" + jsonEscapeString(file1) + ", " + jsonEscapeString(file2) + "],\n";
  json += "    \"covergroups\": [\n";

  bool firstCg = true;
  for (const auto &cg : mergedDb.covergroups) {
    if (!firstCg)
      json += ",\n";
    firstCg = false;

    json += "      {\n";
    json += "        \"name\": \"" + cg.name + "\",\n";
    json += "        \"coverage_percent\": " + std::to_string(cg.coverage) + ",\n";
    json += "        \"num_coverpoints\": " + std::to_string(cg.coverpoints.size()) + ",\n";
    json += "        \"coverpoints\": [\n";

    bool firstCp = true;
    for (const auto &cp : cg.coverpoints) {
      if (!firstCp)
        json += ",\n";
      firstCp = false;

      json += "          {\n";
      json += "            \"name\": \"" + cp.name + "\",\n";
      json += "            \"coverage_percent\": " + std::to_string(cp.coverage) + ",\n";
      json += "            \"total_hits\": " + std::to_string(cp.hits) + ",\n";
      json += "            \"min_value\": " + std::to_string(cp.minVal) + ",\n";
      json += "            \"max_value\": " + std::to_string(cp.maxVal) + ",\n";

      json += "            \"bins\": [\n";
      bool firstBin = true;
      for (const auto &bin : cp.bins) {
        if (!firstBin)
          json += ",\n";
        firstBin = false;

        const char *binTypeName = "unknown";
        switch (bin.type) {
        case MOORE_BIN_VALUE: binTypeName = "value"; break;
        case MOORE_BIN_RANGE: binTypeName = "range"; break;
        case MOORE_BIN_WILDCARD: binTypeName = "wildcard"; break;
        case MOORE_BIN_TRANSITION: binTypeName = "transition"; break;
        }

        json += "              {\n";
        json += "                \"name\": " +
                (bin.name ? ("\"" + std::string(bin.name) + "\"") : "null") + ",\n";
        json += "                \"type\": \"" + std::string(binTypeName) + "\",\n";
        json += "                \"low\": " + std::to_string(bin.low) + ",\n";
        json += "                \"high\": " + std::to_string(bin.high) + ",\n";
        json += "                \"hit_count\": " + std::to_string(bin.hit_count) + "\n";
        json += "              }";
      }
      json += "\n            ],\n";

      json += "            \"top_values\": [\n";
      bool firstVal = true;
      for (const auto &kv : cp.valueCounts) {
        if (!firstVal)
          json += ",\n";
        firstVal = false;
        json += "              {\"value\": " + std::to_string(kv.first) +
                ", \"count\": " + std::to_string(kv.second) + "}";
      }
      json += "\n            ]\n";
      json += "          }";
    }

    json += "\n        ]\n";
    json += "      }";
  }

  json += "\n    ]\n";
  json += "  }\n";
  json += "}\n";

  std::fwrite(json.c_str(), 1, json.size(), fp);
  std::fclose(fp);

  // Clean up merged database bin names
  for (auto &cg : mergedDb.covergroups) {
    for (auto &cp : cg.coverpoints) {
      for (auto &bin : cp.bins) {
        if (bin.name)
          std::free(const_cast<char *>(bin.name));
      }
    }
  }

  __moore_coverage_db_free(db1);
  __moore_coverage_db_free(db2);

  return 0;
}

extern "C" int32_t __moore_coverage_db_get_num_covergroups(MooreCoverageDBHandle db) {
  if (!db)
    return -1;
  return static_cast<int32_t>(db->covergroups.size());
}

extern "C" const char *__moore_coverage_db_get_covergroup_name(MooreCoverageDBHandle db,
                                                                int32_t index) {
  if (!db || index < 0 || index >= static_cast<int32_t>(db->covergroups.size()))
    return nullptr;
  return db->covergroups[index].name.c_str();
}

extern "C" double __moore_coverage_db_get_coverage(MooreCoverageDBHandle db,
                                                    const char *cg_name) {
  if (!db)
    return -1.0;

  if (!cg_name) {
    // Return total coverage
    return db->getTotalCoverage();
  }

  // Find covergroup by name
  for (const auto &cg : db->covergroups) {
    if (cg.name == cg_name)
      return cg.coverage;
  }

  return -1.0;
}

//===----------------------------------------------------------------------===//
// Coverage Database Persistence with Metadata
//===----------------------------------------------------------------------===//
//
// Enhanced coverage database functions that include metadata such as test name,
// timestamp, and other information for tracking coverage across multiple runs.
//

namespace {

/// Generate a complete coverage JSON with metadata.
/// This is used for save_db operations where we need all the data plus metadata.
std::string generateFullCoverageJsonWithMetadata(const char *testName,
                                                  const char *comment) {
  std::string json = "{\n";
  json += "  \"coverage_report\": {\n";
  json += "    \"version\": \"1.1\",\n";
  json += "    \"generator\": \"circt-moore-runtime\",\n";
  json += "    \"format\": \"coverage_database\",\n";

  // Add metadata section
  json += "    \"metadata\": {\n";

  // Test name
  if (testName && testName[0] != '\0') {
    json += "      \"test_name\": " + jsonEscapeString(testName) + ",\n";
  } else if (!globalTestName.empty()) {
    json += "      \"test_name\": " + jsonEscapeString(globalTestName.c_str()) +
            ",\n";
  } else {
    json += "      \"test_name\": null,\n";
  }

  // Timestamp (current time in seconds since epoch)
  int64_t timestamp = static_cast<int64_t>(std::time(nullptr));
  json += "      \"timestamp\": " + std::to_string(timestamp) + ",\n";

  // Simulator name
  json += "      \"simulator\": \"circt-moore\",\n";

  // Comment
  if (comment && comment[0] != '\0') {
    json += "      \"comment\": " + jsonEscapeString(comment) + "\n";
  } else {
    json += "      \"comment\": null\n";
  }

  json += "    },\n";

  json += "    \"covergroups\": [\n";

  bool firstCg = true;
  for (auto *cg : registeredCovergroups) {
    if (!cg)
      continue;

    if (!firstCg)
      json += ",\n";
    firstCg = false;

    double cgCoverage = __moore_covergroup_get_coverage(cg);

    json += "      {\n";
    json += "        \"name\": " + jsonEscapeString(cg->name) + ",\n";
    json +=
        "        \"coverage_percent\": " + std::to_string(cgCoverage) + ",\n";
    json += "        \"num_coverpoints\": " +
            std::to_string(cg->num_coverpoints) + ",\n";
    json += "        \"coverpoints\": [\n";

    bool firstCp = true;
    for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
      auto *cp = cg->coverpoints[i];
      if (!cp)
        continue;

      if (!firstCp)
        json += ",\n";
      firstCp = false;

      double cpCoverage = __moore_coverpoint_get_coverage(cg, i);
      auto trackerIt = coverpointTrackers.find(cp);

      json += "          {\n";
      json += "            \"name\": " + jsonEscapeString(cp->name) + ",\n";
      json += "            \"coverage_percent\": " +
              std::to_string(cpCoverage) + ",\n";
      json += "            \"total_hits\": " + std::to_string(cp->hits) + ",\n";
      json +=
          "            \"min_value\": " + std::to_string(cp->min_val) + ",\n";
      json +=
          "            \"max_value\": " + std::to_string(cp->max_val) + ",\n";

      // Add explicit bins if present
      json += "            \"bins\": [\n";
      auto binIt = explicitBinData.find(cp);
      if (binIt != explicitBinData.end()) {
        bool firstBin = true;
        for (const auto &bin : binIt->second.bins) {
          if (!firstBin)
            json += ",\n";
          firstBin = false;

          const char *binTypeName = "unknown";
          switch (bin.type) {
          case MOORE_BIN_VALUE:
            binTypeName = "value";
            break;
          case MOORE_BIN_RANGE:
            binTypeName = "range";
            break;
          case MOORE_BIN_WILDCARD:
            binTypeName = "wildcard";
            break;
          case MOORE_BIN_TRANSITION:
            binTypeName = "transition";
            break;
          }

          json += "              {\n";
          json += "                \"name\": " + jsonEscapeString(bin.name) +
                  ",\n";
          json += "                \"type\": \"" + std::string(binTypeName) +
                  "\",\n";
          json += "                \"low\": " + std::to_string(bin.low) + ",\n";
          json +=
              "                \"high\": " + std::to_string(bin.high) + ",\n";
          json += "                \"hit_count\": " +
                  std::to_string(bin.hit_count) + "\n";
          json += "              }";
        }
      }
      json += "\n            ],\n";

      // Add ALL value counts (not just top 10) for accurate merging
      json += "            \"top_values\": [\n";
      if (trackerIt != coverpointTrackers.end()) {
        bool firstVal = true;
        for (const auto &kv : trackerIt->second.valueCounts) {
          if (!firstVal)
            json += ",\n";
          firstVal = false;
          json += "              {\"value\": " + std::to_string(kv.first) +
                  ", \"count\": " + std::to_string(kv.second) + "}";
        }
      }
      json += "\n            ]\n";
      json += "          }";
    }

    json += "\n        ]\n";
    json += "      }";
  }

  json += "\n    ]\n";
  json += "  }\n";
  json += "}\n";

  return json;
}

} // anonymous namespace

extern "C" int32_t __moore_coverage_save_db(const char *filename,
                                             const char *test_name,
                                             const char *comment) {
  if (!filename)
    return 1;

  FILE *fp = std::fopen(filename, "w");
  if (!fp)
    return 1;

  std::string json = generateFullCoverageJsonWithMetadata(test_name, comment);
  std::fwrite(json.c_str(), 1, json.size(), fp);
  std::fclose(fp);

  return 0;
}

extern "C" MooreCoverageDBHandle __moore_coverage_load_db(const char *filename) {
  if (!filename)
    return nullptr;

  FILE *fp = std::fopen(filename, "r");
  if (!fp)
    return nullptr;

  // Read the entire file
  std::fseek(fp, 0, SEEK_END);
  long fileSize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  if (fileSize <= 0) {
    std::fclose(fp);
    return nullptr;
  }

  std::string json(fileSize, '\0');
  size_t bytesRead = std::fread(&json[0], 1, fileSize, fp);
  std::fclose(fp);

  if (bytesRead != static_cast<size_t>(fileSize))
    return nullptr;

  // Parse the JSON using the existing parser
  auto *db = new MooreCoverageDB();
  SimpleJsonParser parser(json);
  if (!parser.parse(*db)) {
    delete db;
    return nullptr;
  }

  // Parse metadata manually since it's a new field
  // Look for "metadata" section in the JSON
  size_t metadataPos = json.find("\"metadata\"");
  if (metadataPos != std::string::npos) {
    db->hasMetadata = true;

    // Find test_name
    size_t testNamePos = json.find("\"test_name\"", metadataPos);
    if (testNamePos != std::string::npos) {
      size_t colonPos = json.find(':', testNamePos);
      if (colonPos != std::string::npos) {
        size_t valueStart = json.find_first_not_of(" \t\n\r", colonPos + 1);
        if (valueStart != std::string::npos && json[valueStart] == '"') {
          size_t valueEnd = json.find('"', valueStart + 1);
          if (valueEnd != std::string::npos) {
            db->metadataStorage.testName =
                json.substr(valueStart + 1, valueEnd - valueStart - 1);
          }
        }
      }
    }

    // Find timestamp
    size_t timestampPos = json.find("\"timestamp\"", metadataPos);
    if (timestampPos != std::string::npos) {
      size_t colonPos = json.find(':', timestampPos);
      if (colonPos != std::string::npos) {
        size_t valueStart = json.find_first_not_of(" \t\n\r", colonPos + 1);
        if (valueStart != std::string::npos) {
          db->metadataStorage.timestamp = std::stoll(json.substr(valueStart));
        }
      }
    }

    // Find simulator
    size_t simulatorPos = json.find("\"simulator\"", metadataPos);
    if (simulatorPos != std::string::npos) {
      size_t colonPos = json.find(':', simulatorPos);
      if (colonPos != std::string::npos) {
        size_t valueStart = json.find_first_not_of(" \t\n\r", colonPos + 1);
        if (valueStart != std::string::npos && json[valueStart] == '"') {
          size_t valueEnd = json.find('"', valueStart + 1);
          if (valueEnd != std::string::npos) {
            db->metadataStorage.simulator =
                json.substr(valueStart + 1, valueEnd - valueStart - 1);
          }
        }
      }
    }

    // Find comment
    size_t commentPos = json.find("\"comment\"", metadataPos);
    if (commentPos != std::string::npos) {
      size_t colonPos = json.find(':', commentPos);
      if (colonPos != std::string::npos) {
        size_t valueStart = json.find_first_not_of(" \t\n\r", colonPos + 1);
        if (valueStart != std::string::npos && json[valueStart] == '"') {
          size_t valueEnd = json.find('"', valueStart + 1);
          if (valueEnd != std::string::npos) {
            db->metadataStorage.comment =
                json.substr(valueStart + 1, valueEnd - valueStart - 1);
          }
        }
      }
    }

    // Find version (in metadata section)
    // Note: version is also at the top level, we look for it in metadata first
    db->metadataStorage.version = "1.1"; // Default version for new format

    db->updateMetadataPointers();
  }

  return db;
}

extern "C" int32_t __moore_coverage_merge_db(const char *filename) {
  MooreCoverageDBHandle db = __moore_coverage_load_db(filename);
  if (!db)
    return 1;

  int32_t result = __moore_coverage_merge(db);
  __moore_coverage_db_free(db);
  return result;
}

extern "C" const MooreCoverageMetadata *
__moore_coverage_db_get_metadata(MooreCoverageDBHandle db) {
  if (!db || !db->hasMetadata)
    return nullptr;
  return &db->metadata;
}

extern "C" void __moore_coverage_set_test_name(const char *test_name) {
  if (test_name)
    globalTestName = test_name;
  else
    globalTestName.clear();
}

extern "C" const char *__moore_coverage_get_test_name(void) {
  return globalTestName.empty() ? nullptr : globalTestName.c_str();
}

//===----------------------------------------------------------------------===//
// UCDB-Compatible Coverage File Format
//===----------------------------------------------------------------------===//
//
// These functions implement UCDB (Unified Coverage Database) compatible file
// format support. The format uses JSON for human readability and portability,
// with a schema designed to map to UCDB semantics.
//

namespace {

/// Global user attributes for coverage sessions.
std::map<std::string, std::string> globalUserAttrs;

/// Thread-local storage for version string return value.
thread_local std::string cachedVersionString;

/// Get hostname for metadata.
std::string getHostnameStr() {
#ifdef _WIN32
  char hostname[256];
  DWORD size = sizeof(hostname);
  if (GetComputerNameA(hostname, &size))
    return hostname;
  return "unknown";
#else
  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) == 0)
    return hostname;
  return "unknown";
#endif
}

/// Get username for metadata.
std::string getUsernameStr() {
#ifdef _WIN32
  char username[256];
  DWORD size = sizeof(username);
  if (GetUserNameA(username, &size))
    return username;
  return "unknown";
#else
  const char *user = getenv("USER");
  if (user)
    return user;
  user = getenv("LOGNAME");
  if (user)
    return user;
  return "unknown";
#endif
}

/// Get current working directory for metadata.
std::string getCurrentWorkdirStr() {
  char cwd[4096];
#ifdef _WIN32
  if (_getcwd(cwd, sizeof(cwd)))
    return cwd;
#else
  if (getcwd(cwd, sizeof(cwd)))
    return cwd;
#endif
  return "";
}

/// Generate UCDB-compatible JSON output.
std::string generateUCDBJson(const MooreUCDBMetadata *metadata) {
  std::string json = "{\n";

  // Schema identification
  json += "  \"$schema\": \"circt-ucdb-2.0\",\n";
  json += "  \"format\": {\n";
  json += "    \"name\": \"" MOORE_UCDB_FORMAT_MAGIC "\",\n";
  json += "    \"version\": \"" MOORE_UCDB_FORMAT_VERSION "\",\n";
  json += "    \"generator\": \"circt-moore-runtime\",\n";
  json += "    \"generator_version\": \"1.0.0\"\n";
  json += "  },\n";

  // Metadata section
  json += "  \"metadata\": {\n";

  // Test info
  if (metadata && metadata->test_name) {
    json +=
        "    \"test_name\": " + jsonEscapeString(metadata->test_name) + ",\n";
  } else if (!globalTestName.empty()) {
    json += "    \"test_name\": " +
            jsonEscapeString(globalTestName.c_str()) + ",\n";
  } else {
    json += "    \"test_name\": null,\n";
  }

  if (metadata && metadata->test_seed) {
    json +=
        "    \"test_seed\": " + jsonEscapeString(metadata->test_seed) + ",\n";
  } else {
    json += "    \"test_seed\": null,\n";
  }

  // Timestamps
  int64_t currentTime = static_cast<int64_t>(std::time(nullptr));
  if (metadata && metadata->start_time > 0) {
    json +=
        "    \"start_time\": " + std::to_string(metadata->start_time) + ",\n";
  } else {
    json += "    \"start_time\": " + std::to_string(currentTime) + ",\n";
  }

  if (metadata && metadata->end_time > 0) {
    json += "    \"end_time\": " + std::to_string(metadata->end_time) + ",\n";
  } else {
    json += "    \"end_time\": " + std::to_string(currentTime) + ",\n";
  }

  if (metadata && metadata->sim_time > 0) {
    json += "    \"sim_time\": " + std::to_string(metadata->sim_time) + ",\n";
  } else {
    json += "    \"sim_time\": 0,\n";
  }

  if (metadata && metadata->time_unit) {
    json +=
        "    \"time_unit\": " + jsonEscapeString(metadata->time_unit) + ",\n";
  } else {
    json += "    \"time_unit\": \"ns\",\n";
  }

  // Tool info
  if (metadata && metadata->tool_name) {
    json +=
        "    \"tool_name\": " + jsonEscapeString(metadata->tool_name) + ",\n";
  } else {
    json += "    \"tool_name\": \"circt-moore\",\n";
  }

  if (metadata && metadata->tool_version) {
    json += "    \"tool_version\": " +
            jsonEscapeString(metadata->tool_version) + ",\n";
  } else {
    json += "    \"tool_version\": \"1.0.0\",\n";
  }

  // Environment info
  json += "    \"hostname\": " +
          jsonEscapeString(getHostnameStr().c_str()) + ",\n";
  json += "    \"username\": " +
          jsonEscapeString(getUsernameStr().c_str()) + ",\n";

  std::string workdir = getCurrentWorkdirStr();
  if (metadata && metadata->workdir) {
    json += "    \"workdir\": " + jsonEscapeString(metadata->workdir) + ",\n";
  } else if (!workdir.empty()) {
    json += "    \"workdir\": " + jsonEscapeString(workdir.c_str()) + ",\n";
  } else {
    json += "    \"workdir\": null,\n";
  }

  if (metadata && metadata->command_line) {
    json += "    \"command_line\": " +
            jsonEscapeString(metadata->command_line) + ",\n";
  } else {
    json += "    \"command_line\": null,\n";
  }

  if (metadata && metadata->comment) {
    json += "    \"comment\": " + jsonEscapeString(metadata->comment) + ",\n";
  } else {
    json += "    \"comment\": null,\n";
  }

  // Merge history
  json += "    \"merge_history\": [\n";
  if (metadata && metadata->num_merged_runs > 0 &&
      metadata->merged_test_names) {
    for (int32_t i = 0; i < metadata->num_merged_runs; ++i) {
      if (i > 0)
        json += ",\n";
      json += "      " + jsonEscapeString(metadata->merged_test_names[i]);
    }
    if (metadata->num_merged_runs > 0)
      json += "\n";
  }
  json += "    ],\n";

  // User attributes
  json += "    \"user_attributes\": {\n";
  bool firstAttr = true;

  // Add global user attributes
  for (const auto &attr : globalUserAttrs) {
    if (!firstAttr)
      json += ",\n";
    firstAttr = false;
    json += "      " + jsonEscapeString(attr.first.c_str()) + ": " +
            jsonEscapeString(attr.second.c_str());
  }

  // Add metadata user attributes (if any)
  if (metadata && metadata->num_user_attrs > 0 && metadata->user_attr_names &&
      metadata->user_attr_values) {
    for (int32_t i = 0; i < metadata->num_user_attrs; ++i) {
      if (!firstAttr)
        json += ",\n";
      firstAttr = false;
      json += "      " + jsonEscapeString(metadata->user_attr_names[i]) + ": " +
              jsonEscapeString(metadata->user_attr_values[i]);
    }
  }

  if (!firstAttr)
    json += "\n";
  json += "    }\n";
  json += "  },\n";

  // Coverage data section
  json += "  \"coverage_data\": {\n";

  // Summary
  double totalCoverage = 0.0;
  int32_t totalCovergroups = 0;
  int32_t totalCoverpoints = 0;
  int64_t totalBins = 0;
  int64_t coveredBins = 0;

  for (auto *cg : registeredCovergroups) {
    if (!cg)
      continue;
    ++totalCovergroups;
    totalCoverpoints += cg->num_coverpoints;
    totalCoverage += __moore_covergroup_get_coverage(cg);

    for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
      auto *cp = cg->coverpoints[i];
      if (!cp)
        continue;
      auto binIt = explicitBinData.find(cp);
      if (binIt != explicitBinData.end()) {
        totalBins += binIt->second.bins.size();
        for (const auto &bin : binIt->second.bins) {
          if (bin.hit_count > 0)
            ++coveredBins;
        }
      }
    }
  }

  if (totalCovergroups > 0)
    totalCoverage /= totalCovergroups;

  json += "    \"summary\": {\n";
  json += "      \"total_coverage_percent\": " + std::to_string(totalCoverage) +
          ",\n";
  json += "      \"total_covergroups\": " + std::to_string(totalCovergroups) +
          ",\n";
  json += "      \"total_coverpoints\": " + std::to_string(totalCoverpoints) +
          ",\n";
  json += "      \"total_bins\": " + std::to_string(totalBins) + ",\n";
  json += "      \"covered_bins\": " + std::to_string(coveredBins) + "\n";
  json += "    },\n";

  // Covergroups
  json += "    \"covergroups\": [\n";

  bool firstCg = true;
  for (auto *cg : registeredCovergroups) {
    if (!cg)
      continue;

    if (!firstCg)
      json += ",\n";
    firstCg = false;

    double cgCoverage = __moore_covergroup_get_coverage(cg);

    json += "      {\n";
    json += "        \"name\": " + jsonEscapeString(cg->name) + ",\n";
    json +=
        "        \"coverage_percent\": " + std::to_string(cgCoverage) + ",\n";
    json += "        \"type\": \"covergroup\",\n";
    json += "        \"options\": {\n";
    json += "          \"weight\": 1,\n";
    json += "          \"goal\": 100,\n";
    json += "          \"comment\": null\n";
    json += "        },\n";
    json += "        \"coverpoints\": [\n";

    bool firstCp = true;
    for (int32_t i = 0; i < cg->num_coverpoints; ++i) {
      auto *cp = cg->coverpoints[i];
      if (!cp)
        continue;

      if (!firstCp)
        json += ",\n";
      firstCp = false;

      double cpCoverage = __moore_coverpoint_get_coverage(cg, i);
      auto trackerIt = coverpointTrackers.find(cp);

      json += "          {\n";
      json += "            \"name\": " + jsonEscapeString(cp->name) + ",\n";
      json += "            \"coverage_percent\": " +
              std::to_string(cpCoverage) + ",\n";
      json += "            \"total_hits\": " + std::to_string(cp->hits) + ",\n";
      json +=
          "            \"min_value\": " + std::to_string(cp->min_val) + ",\n";
      json +=
          "            \"max_value\": " + std::to_string(cp->max_val) + ",\n";
      json += "            \"options\": {\n";
      json += "              \"weight\": 1,\n";
      json += "              \"goal\": 100,\n";
      json += "              \"at_least\": 1,\n";
      json += "              \"auto_bin_max\": 64,\n";
      json += "              \"comment\": null\n";
      json += "            },\n";

      // Bins
      json += "            \"bins\": [\n";
      auto binIt = explicitBinData.find(cp);
      if (binIt != explicitBinData.end()) {
        bool firstBin = true;
        for (const auto &bin : binIt->second.bins) {
          if (!firstBin)
            json += ",\n";
          firstBin = false;

          const char *binTypeName = "unknown";
          switch (bin.type) {
          case MOORE_BIN_VALUE:
            binTypeName = "value";
            break;
          case MOORE_BIN_RANGE:
            binTypeName = "range";
            break;
          case MOORE_BIN_WILDCARD:
            binTypeName = "wildcard";
            break;
          case MOORE_BIN_TRANSITION:
            binTypeName = "transition";
            break;
          }

          const char *binKindName = "normal";
          switch (bin.kind) {
          case MOORE_BIN_KIND_NORMAL:
            binKindName = "normal";
            break;
          case MOORE_BIN_KIND_ILLEGAL:
            binKindName = "illegal";
            break;
          case MOORE_BIN_KIND_IGNORE:
            binKindName = "ignore";
            break;
          }

          json += "              {\n";
          json +=
              "                \"name\": " + jsonEscapeString(bin.name) + ",\n";
          json += "                \"type\": \"" + std::string(binTypeName) +
                  "\",\n";
          json += "                \"kind\": \"" + std::string(binKindName) +
                  "\",\n";
          json += "                \"low\": " + std::to_string(bin.low) + ",\n";
          json +=
              "                \"high\": " + std::to_string(bin.high) + ",\n";
          json += "                \"hit_count\": " +
                  std::to_string(bin.hit_count) + ",\n";
          json += "                \"at_least\": 1\n";
          json += "              }";
        }
      }
      json += "\n            ],\n";

      // Value distribution (for analysis)
      json += "            \"value_distribution\": [\n";
      if (trackerIt != coverpointTrackers.end()) {
        bool firstVal = true;
        for (const auto &kv : trackerIt->second.valueCounts) {
          if (!firstVal)
            json += ",\n";
          firstVal = false;
          json += "              {\"value\": " + std::to_string(kv.first) +
                  ", \"count\": " + std::to_string(kv.second) + "}";
        }
      }
      json += "\n            ]\n";
      json += "          }";
    }

    json += "\n        ],\n";

    // Crosses placeholder (for future)
    json += "        \"crosses\": []\n";
    json += "      }";
  }

  json += "\n    ]\n";
  json += "  }\n";
  json += "}\n";

  return json;
}

} // anonymous namespace

extern "C" int32_t
__moore_coverage_write_ucdb(const char *filename,
                            const MooreUCDBMetadata *metadata) {
  if (!filename)
    return 1;

  FILE *fp = std::fopen(filename, "w");
  if (!fp)
    return 1;

  std::string json = generateUCDBJson(metadata);
  size_t written = std::fwrite(json.c_str(), 1, json.size(), fp);
  std::fclose(fp);

  return (written == json.size()) ? 0 : 1;
}

extern "C" MooreCoverageDBHandle
__moore_coverage_read_ucdb(const char *filename) {
  if (!filename)
    return nullptr;

  FILE *fp = std::fopen(filename, "r");
  if (!fp)
    return nullptr;

  // Read the entire file
  std::fseek(fp, 0, SEEK_END);
  long fileSize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  if (fileSize <= 0) {
    std::fclose(fp);
    return nullptr;
  }

  std::string json(fileSize, '\0');
  size_t bytesRead = std::fread(&json[0], 1, fileSize, fp);
  std::fclose(fp);

  if (bytesRead != static_cast<size_t>(fileSize))
    return nullptr;

  // Allocate database
  auto *db = new MooreCoverageDB();

  // Use the existing JSON parser
  SimpleJsonParser parser(json);
  if (!parser.parse(*db)) {
    delete db;
    return nullptr;
  }

  // Parse metadata if present
  size_t metadataPos = json.find("\"metadata\"");
  if (metadataPos != std::string::npos) {
    db->hasMetadata = true;

    // Parse test_name
    size_t testNamePos = json.find("\"test_name\"", metadataPos);
    if (testNamePos != std::string::npos) {
      size_t colonPos = json.find(':', testNamePos);
      if (colonPos != std::string::npos) {
        size_t valueStart = json.find_first_not_of(" \t\n\r", colonPos + 1);
        if (valueStart != std::string::npos && json[valueStart] == '"') {
          size_t valueEnd = json.find('"', valueStart + 1);
          if (valueEnd != std::string::npos) {
            db->metadataStorage.testName =
                json.substr(valueStart + 1, valueEnd - valueStart - 1);
          }
        }
      }
    }

    // Parse tool_name as simulator
    size_t toolNamePos = json.find("\"tool_name\"", metadataPos);
    if (toolNamePos != std::string::npos) {
      size_t colonPos = json.find(':', toolNamePos);
      if (colonPos != std::string::npos) {
        size_t valueStart = json.find_first_not_of(" \t\n\r", colonPos + 1);
        if (valueStart != std::string::npos && json[valueStart] == '"') {
          size_t valueEnd = json.find('"', valueStart + 1);
          if (valueEnd != std::string::npos) {
            db->metadataStorage.simulator =
                json.substr(valueStart + 1, valueEnd - valueStart - 1);
          }
        }
      }
    }

    // Parse end_time as timestamp
    size_t endTimePos = json.find("\"end_time\"", metadataPos);
    if (endTimePos != std::string::npos) {
      size_t colonPos = json.find(':', endTimePos);
      if (colonPos != std::string::npos) {
        size_t valueStart = json.find_first_not_of(" \t\n\r", colonPos + 1);
        if (valueStart != std::string::npos) {
          // Parse integer without exceptions
          const char *numStart = json.c_str() + valueStart;
          char *numEnd = nullptr;
          long long value = std::strtoll(numStart, &numEnd, 10);
          db->metadataStorage.timestamp =
              (numEnd != numStart) ? static_cast<int64_t>(value) : 0;
        }
      }
    }

    // Parse comment
    size_t commentPos = json.find("\"comment\"", metadataPos);
    if (commentPos != std::string::npos) {
      size_t colonPos = json.find(':', commentPos);
      if (colonPos != std::string::npos) {
        size_t valueStart = json.find_first_not_of(" \t\n\r", colonPos + 1);
        if (valueStart != std::string::npos && json[valueStart] == '"') {
          size_t valueEnd = json.find('"', valueStart + 1);
          if (valueEnd != std::string::npos) {
            db->metadataStorage.comment =
                json.substr(valueStart + 1, valueEnd - valueStart - 1);
          }
        }
      }
    }

    // Check for UCDB format version
    size_t formatPos = json.find("\"format\"");
    if (formatPos != std::string::npos) {
      size_t versionPos = json.find("\"version\"", formatPos);
      if (versionPos != std::string::npos && versionPos < formatPos + 200) {
        size_t colonPos = json.find(':', versionPos);
        if (colonPos != std::string::npos) {
          size_t valueStart = json.find_first_not_of(" \t\n\r", colonPos + 1);
          if (valueStart != std::string::npos && json[valueStart] == '"') {
            size_t valueEnd = json.find('"', valueStart + 1);
            if (valueEnd != std::string::npos) {
              db->metadataStorage.version =
                  json.substr(valueStart + 1, valueEnd - valueStart - 1);
            }
          }
        }
      }
    }

    db->updateMetadataPointers();
  }

  return db;
}

extern "C" const MooreUCDBMetadata *
__moore_coverage_db_get_ucdb_metadata(MooreCoverageDBHandle db) {
  // Extended UCDB metadata is not currently stored in the handle
  // Return nullptr - full implementation would need extended type
  if (!db)
    return nullptr;
  return nullptr;
}

extern "C" int32_t __moore_coverage_merge_ucdb_files(const char **input_files,
                                                      int32_t num_files,
                                                      const char *output_file,
                                                      const char *comment) {
  if (!input_files || num_files <= 0 || !output_file)
    return 1;

  // Load all input databases
  std::vector<MooreCoverageDBHandle> databases;
  std::vector<std::string> mergedTestNames;

  for (int32_t i = 0; i < num_files; ++i) {
    if (!input_files[i])
      continue;

    MooreCoverageDBHandle db = __moore_coverage_read_ucdb(input_files[i]);
    if (!db) {
      // Clean up already loaded databases
      for (auto *d : databases)
        __moore_coverage_db_free(d);
      return 1;
    }

    databases.push_back(db);

    // Collect test names for merge history
    const MooreCoverageMetadata *meta = __moore_coverage_db_get_metadata(db);
    if (meta && meta->test_name) {
      mergedTestNames.push_back(meta->test_name);
    } else {
      mergedTestNames.push_back(input_files[i]);
    }
  }

  if (databases.empty())
    return 1;

  // Create merged database starting with first file's data
  MooreCoverageDB mergedDb;
  for (const auto &cg : databases[0]->covergroups) {
    mergedDb.covergroups.push_back(cg);
  }

  // Merge remaining databases
  for (size_t i = 1; i < databases.size(); ++i) {
    for (const auto &cg : databases[i]->covergroups) {
      bool found = false;
      for (auto &mergedCg : mergedDb.covergroups) {
        if (mergedCg.name == cg.name) {
          // Merge coverpoints
          for (const auto &cp : cg.coverpoints) {
            bool cpFound = false;
            for (auto &mergedCp : mergedCg.coverpoints) {
              if (mergedCp.name == cp.name) {
                mergedCp.hits += cp.hits;
                mergedCp.minVal = std::min(mergedCp.minVal, cp.minVal);
                mergedCp.maxVal = std::max(mergedCp.maxVal, cp.maxVal);
                for (const auto &vc : cp.valueCounts) {
                  mergedCp.valueCounts[vc.first] += vc.second;
                }
                for (size_t binIdx = 0;
                     binIdx < cp.bins.size() && binIdx < mergedCp.bins.size();
                     ++binIdx) {
                  mergedCp.bins[binIdx].hit_count += cp.bins[binIdx].hit_count;
                }
                cpFound = true;
                break;
              }
            }
            if (!cpFound) {
              mergedCg.coverpoints.push_back(cp);
            }
          }
          found = true;
          break;
        }
      }
      if (!found) {
        mergedDb.covergroups.push_back(cg);
      }
    }
  }

  // Clean up input databases
  for (auto *db : databases)
    __moore_coverage_db_free(db);

  // Write merged output
  FILE *fp = std::fopen(output_file, "w");
  if (!fp)
    return 1;

  // Generate UCDB-format JSON for merged data
  std::string json = "{\n";
  json += "  \"$schema\": \"circt-ucdb-2.0\",\n";
  json += "  \"format\": {\n";
  json += "    \"name\": \"" MOORE_UCDB_FORMAT_MAGIC "\",\n";
  json += "    \"version\": \"" MOORE_UCDB_FORMAT_VERSION "\",\n";
  json += "    \"generator\": \"circt-moore-runtime\"\n";
  json += "  },\n";

  json += "  \"metadata\": {\n";
  json += "    \"test_name\": \"merged_coverage\",\n";
  json += "    \"end_time\": " +
          std::to_string(static_cast<int64_t>(std::time(nullptr))) + ",\n";
  json += "    \"tool_name\": \"circt-moore\",\n";
  if (comment) {
    json += "    \"comment\": " + jsonEscapeString(comment) + ",\n";
  } else {
    json += "    \"comment\": null,\n";
  }
  json += "    \"merge_history\": [\n";
  for (size_t i = 0; i < mergedTestNames.size(); ++i) {
    if (i > 0)
      json += ",\n";
    json += "      " + jsonEscapeString(mergedTestNames[i].c_str());
  }
  json += "\n    ]\n";
  json += "  },\n";

  // Coverage data
  json += "  \"coverage_data\": {\n";
  json += "    \"covergroups\": [\n";
  bool firstCg = true;
  for (const auto &cg : mergedDb.covergroups) {
    if (!firstCg)
      json += ",\n";
    firstCg = false;

    json += "      {\n";
    json += "        \"name\": " + jsonEscapeString(cg.name.c_str()) + ",\n";
    json +=
        "        \"coverage_percent\": " + std::to_string(cg.coverage) + ",\n";
    json += "        \"coverpoints\": [\n";

    bool firstCp = true;
    for (const auto &cp : cg.coverpoints) {
      if (!firstCp)
        json += ",\n";
      firstCp = false;

      json += "          {\n";
      json +=
          "            \"name\": " + jsonEscapeString(cp.name.c_str()) + ",\n";
      json += "            \"total_hits\": " + std::to_string(cp.hits) + ",\n";
      json +=
          "            \"min_value\": " + std::to_string(cp.minVal) + ",\n";
      json +=
          "            \"max_value\": " + std::to_string(cp.maxVal) + ",\n";
      json += "            \"bins\": [\n";

      bool firstBin = true;
      for (const auto &bin : cp.bins) {
        if (!firstBin)
          json += ",\n";
        firstBin = false;
        json += "              {\"name\": " + jsonEscapeString(bin.name) +
                ", \"hit_count\": " + std::to_string(bin.hit_count) + "}";
      }
      json += "\n            ],\n";

      json += "            \"top_values\": [\n";
      bool firstVal = true;
      for (const auto &kv : cp.valueCounts) {
        if (!firstVal)
          json += ",\n";
        firstVal = false;
        json += "              {\"value\": " + std::to_string(kv.first) +
                ", \"count\": " + std::to_string(kv.second) + "}";
      }
      json += "\n            ]\n";
      json += "          }";
    }

    json += "\n        ]\n";
    json += "      }";
  }

  json += "\n    ]\n";
  json += "  }\n";
  json += "}\n";

  size_t written = std::fwrite(json.c_str(), 1, json.size(), fp);
  std::fclose(fp);

  return (written == json.size()) ? 0 : 1;
}

extern "C" int32_t __moore_coverage_is_ucdb_format(const char *filename) {
  if (!filename)
    return -1;

  FILE *fp = std::fopen(filename, "r");
  if (!fp)
    return -1;

  char buffer[1024];
  size_t bytesRead = std::fread(buffer, 1, sizeof(buffer) - 1, fp);
  std::fclose(fp);

  if (bytesRead == 0)
    return -1;

  buffer[bytesRead] = '\0';
  std::string header(buffer);

  // Check for UCDB-specific markers (more specific than generic "format")
  // - "$schema" is a JSON schema indicator unique to UCDB 2.0
  // - "circt-ucdb" is our format identifier magic string
  // - "format": { indicates a format object (UCDB) vs "format": "string" (legacy)
  if (header.find("\"$schema\"") != std::string::npos ||
      header.find("circt-ucdb") != std::string::npos) {
    return 1;
  }

  // Check for UCDB format object structure (has "format": { with "name" inside)
  size_t formatPos = header.find("\"format\"");
  if (formatPos != std::string::npos) {
    size_t colonPos = header.find(':', formatPos);
    if (colonPos != std::string::npos) {
      size_t valueStart = header.find_first_not_of(" \t\n\r", colonPos + 1);
      // If format value starts with '{', it's UCDB format
      if (valueStart != std::string::npos && header[valueStart] == '{') {
        return 1;
      }
    }
  }

  return 0;
}

extern "C" const char *__moore_coverage_get_file_version(const char *filename) {
  if (!filename)
    return nullptr;

  FILE *fp = std::fopen(filename, "r");
  if (!fp)
    return nullptr;

  char buffer[2048];
  size_t bytesRead = std::fread(buffer, 1, sizeof(buffer) - 1, fp);
  std::fclose(fp);

  if (bytesRead == 0)
    return nullptr;

  buffer[bytesRead] = '\0';
  std::string content(buffer);

  // Look for version in format section (UCDB 2.0+)
  size_t formatPos = content.find("\"format\"");
  if (formatPos != std::string::npos) {
    size_t versionPos = content.find("\"version\"", formatPos);
    if (versionPos != std::string::npos && versionPos < formatPos + 200) {
      size_t colonPos = content.find(':', versionPos);
      if (colonPos != std::string::npos) {
        size_t valueStart = content.find_first_not_of(" \t\n\r", colonPos + 1);
        if (valueStart != std::string::npos && content[valueStart] == '"') {
          size_t valueEnd = content.find('"', valueStart + 1);
          if (valueEnd != std::string::npos) {
            cachedVersionString =
                content.substr(valueStart + 1, valueEnd - valueStart - 1);
            return cachedVersionString.c_str();
          }
        }
      }
    }
  }

  // Try legacy version field
  size_t versionPos = content.find("\"version\"");
  if (versionPos != std::string::npos) {
    size_t colonPos = content.find(':', versionPos);
    if (colonPos != std::string::npos) {
      size_t valueStart = content.find_first_not_of(" \t\n\r", colonPos + 1);
      if (valueStart != std::string::npos) {
        if (content[valueStart] == '"') {
          size_t valueEnd = content.find('"', valueStart + 1);
          if (valueEnd != std::string::npos) {
            cachedVersionString =
                content.substr(valueStart + 1, valueEnd - valueStart - 1);
            return cachedVersionString.c_str();
          }
        }
      }
    }
  }

  return nullptr;
}

extern "C" void __moore_coverage_set_user_attr(const char *name,
                                                const char *value) {
  if (!name)
    return;
  if (value) {
    globalUserAttrs[name] = value;
  } else {
    globalUserAttrs.erase(name);
  }
}

extern "C" const char *__moore_coverage_get_user_attr(const char *name) {
  if (!name)
    return nullptr;
  auto it = globalUserAttrs.find(name);
  if (it != globalUserAttrs.end()) {
    return it->second.c_str();
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Constraint Solving Operations
//===----------------------------------------------------------------------===//
//
// These functions provide basic constraint-aware randomization support.
// They are placeholder stubs for future Z3/SMT solver integration.
// Currently they implement simple bounds-based randomization.
//

extern "C" int __moore_constraint_check_range(int64_t value, int64_t min,
                                               int64_t max) {
  // Check if value is within the inclusive range [min, max].
  // Returns 1 if the constraint is satisfied, 0 otherwise.
  return (value >= min && value <= max) ? 1 : 0;
}

extern "C" int64_t __moore_randomize_with_range(int64_t min, int64_t max) {
  // Generate a random value within the range [min, max].
  // Handle edge cases where min >= max.
  if (min >= max)
    return min;

  // Use the existing random generator for consistency.
  // Calculate the range size carefully to avoid overflow.
  uint64_t range = static_cast<uint64_t>(max - min) + 1;

  // Generate a random value in [0, range-1] and add min.
  // For ranges larger than UINT32_MAX, we need multiple random calls,
  // but for now we use a single 32-bit random value which covers most cases.
  uint64_t randomVal = __moore_urandom();

  // For large ranges, combine two 32-bit values to get 64 bits
  if (range > UINT32_MAX) {
    randomVal = (static_cast<uint64_t>(__moore_urandom()) << 32) |
                __moore_urandom();
  }

  return min + static_cast<int64_t>(randomVal % range);
}

extern "C" int64_t __moore_randomize_with_modulo(int64_t mod, int64_t remainder) {
  // Generate a random value that satisfies: value % mod == remainder.
  // For basic implementation, we generate a random base value and compute
  // base * mod + remainder.
  //
  // Note: This is a simplified implementation. A full constraint solver
  // would handle more complex modulo constraints in combination with
  // other constraints.

  // Handle invalid inputs
  if (mod <= 0)
    return remainder; // Invalid modulo, just return the remainder

  // Normalize remainder to be in valid range [0, mod-1]
  int64_t normalizedRemainder = remainder % mod;
  if (normalizedRemainder < 0)
    normalizedRemainder += mod;

  // Generate a random multiplier and compute the result.
  // We use a 32-bit random value for the multiplier to avoid overflow
  // while still providing reasonable randomness.
  uint32_t multiplier = __moore_urandom() & 0x7FFFFFFF; // Keep positive

  // Compute result = multiplier * mod + normalizedRemainder
  // This guarantees result % mod == normalizedRemainder
  return static_cast<int64_t>(multiplier) * mod + normalizedRemainder;
}

extern "C" int64_t __moore_randomize_with_ranges(int64_t *ranges,
                                                  int64_t numRanges) {
  // Generate a random value that falls within one of the given ranges.
  // The ranges array contains pairs of [min1, max1, min2, max2, ...].
  // numRanges is the number of range pairs (not the array length).
  //
  // Algorithm:
  // 1. Calculate the total size of all ranges combined
  // 2. Generate a random position within the total size
  // 3. Map that position to a specific range and value within it
  //
  // This ensures uniform distribution across all ranges.

  // Validate inputs
  if (!ranges || numRanges <= 0)
    return 0;

  // Calculate total size across all ranges
  uint64_t totalSize = 0;
  for (int64_t i = 0; i < numRanges; ++i) {
    int64_t low = ranges[i * 2];
    int64_t high = ranges[i * 2 + 1];
    // Handle inverted ranges (swap if needed)
    if (low > high) {
      int64_t tmp = low;
      low = high;
      high = tmp;
    }
    // Size of this range: high - low + 1
    uint64_t rangeSize = static_cast<uint64_t>(high - low) + 1;
    totalSize += rangeSize;
  }

  // If total size is 0 (shouldn't happen), return first range's min
  if (totalSize == 0)
    return ranges[0];

  // Generate random position in [0, totalSize - 1]
  uint64_t randomVal = __moore_urandom();
  // For large total sizes, combine two 32-bit values
  if (totalSize > UINT32_MAX) {
    randomVal = (static_cast<uint64_t>(__moore_urandom()) << 32) |
                __moore_urandom();
  }
  uint64_t position = randomVal % totalSize;

  // Map position to a specific range and value
  uint64_t accumulated = 0;
  for (int64_t i = 0; i < numRanges; ++i) {
    int64_t low = ranges[i * 2];
    int64_t high = ranges[i * 2 + 1];
    // Handle inverted ranges
    if (low > high) {
      int64_t tmp = low;
      low = high;
      high = tmp;
    }
    uint64_t rangeSize = static_cast<uint64_t>(high - low) + 1;

    // Check if position falls within this range
    if (position < accumulated + rangeSize) {
      // Position is in this range
      uint64_t offset = position - accumulated;
      return low + static_cast<int64_t>(offset);
    }
    accumulated += rangeSize;
  }

  // Fallback (shouldn't reach here)
  return ranges[0];
}

//===----------------------------------------------------------------------===//
// Constraint Solving with Iteration Limits
//===----------------------------------------------------------------------===//
//
// These functions implement constraint-aware randomization with iteration
// limits and fallback strategies. They prevent infinite loops on unsatisfiable
// constraints and provide diagnostics for debugging.
//

namespace {

/// Global constraint solving configuration and statistics.
struct ConstraintSolverState {
  std::atomic<int64_t> iterationLimit{MOORE_CONSTRAINT_DEFAULT_ITERATION_LIMIT};
  std::atomic<bool> warningsEnabled{true};

  // Statistics (accessed atomically)
  std::atomic<int64_t> totalAttempts{0};
  std::atomic<int64_t> successfulSolves{0};
  std::atomic<int64_t> fallbackCount{0};
  std::atomic<int64_t> iterationLimitHits{0};

  // Last solve info (thread-local for accuracy)
  int64_t lastIterations{0};
};

/// Global constraint solver state.
ConstraintSolverState constraintState;

/// Thread-local statistics for the last operation.
thread_local MooreConstraintStats localStats = {0, 0, 0, 0, 0};

} // anonymous namespace

extern "C" MooreConstraintStats *__moore_constraint_get_stats(void) {
  // Update local stats from global atomics
  localStats.totalAttempts = constraintState.totalAttempts.load();
  localStats.successfulSolves = constraintState.successfulSolves.load();
  localStats.fallbackCount = constraintState.fallbackCount.load();
  localStats.iterationLimitHits = constraintState.iterationLimitHits.load();
  // lastIterations is thread-local, keep as is
  return &localStats;
}

extern "C" void __moore_constraint_reset_stats(void) {
  constraintState.totalAttempts.store(0);
  constraintState.successfulSolves.store(0);
  constraintState.fallbackCount.store(0);
  constraintState.iterationLimitHits.store(0);
  localStats = {0, 0, 0, 0, 0};
}

extern "C" void __moore_constraint_set_iteration_limit(int64_t limit) {
  if (limit <= 0) {
    constraintState.iterationLimit.store(MOORE_CONSTRAINT_DEFAULT_ITERATION_LIMIT);
  } else {
    constraintState.iterationLimit.store(limit);
  }
}

extern "C" int64_t __moore_constraint_get_iteration_limit(void) {
  return constraintState.iterationLimit.load();
}

extern "C" void __moore_constraint_set_warnings_enabled(bool enabled) {
  constraintState.warningsEnabled.store(enabled);
}

extern "C" bool __moore_constraint_warnings_enabled(void) {
  return constraintState.warningsEnabled.load();
}

extern "C" void __moore_constraint_warn(const char *message, int64_t iterations,
                                         const char *variableName) {
  if (!constraintState.warningsEnabled.load())
    return;

  std::fprintf(stderr, "** Warning: Constraint solving issue: %s\n", message);
  if (variableName && variableName[0] != '\0') {
    std::fprintf(stderr, "   Variable: %s\n", variableName);
  }
  std::fprintf(stderr, "   Iterations attempted: %ld\n",
               static_cast<long>(iterations));
  std::fprintf(stderr, "   Using fallback random value.\n");
}

extern "C" int64_t __moore_randomize_with_constraint(int64_t min, int64_t max,
                                                      MooreConstraintPredicate predicate,
                                                      void *userData,
                                                      int64_t iterationLimit,
                                                      int32_t *resultOut) {
  // Update statistics
  constraintState.totalAttempts.fetch_add(1);

  // Determine effective iteration limit
  int64_t limit = iterationLimit > 0 ? iterationLimit
                                     : constraintState.iterationLimit.load();

  // Handle edge case where min >= max
  if (min > max) {
    int64_t tmp = min;
    min = max;
    max = tmp;
  }

  // If no predicate, just do simple range randomization
  if (!predicate) {
    constraintState.successfulSolves.fetch_add(1);
    localStats.lastIterations = 1;
    if (resultOut)
      *resultOut = MOORE_CONSTRAINT_SUCCESS;
    return __moore_randomize_with_range(min, max);
  }

  // Try to find a value that satisfies the predicate
  uint64_t range = static_cast<uint64_t>(max - min) + 1;
  int64_t iterations = 0;

  for (iterations = 0; iterations < limit; ++iterations) {
    // Generate random value in range
    int64_t value = __moore_randomize_with_range(min, max);

    // Check if it satisfies the predicate
    if (predicate(value, userData)) {
      constraintState.successfulSolves.fetch_add(1);
      localStats.lastIterations = iterations + 1;
      if (resultOut)
        *resultOut = MOORE_CONSTRAINT_SUCCESS;
      return value;
    }
  }

  // Hit iteration limit - use fallback
  constraintState.iterationLimitHits.fetch_add(1);
  constraintState.fallbackCount.fetch_add(1);
  localStats.lastIterations = iterations;

  // Emit warning
  __moore_constraint_warn("Hit iteration limit, constraint may be unsatisfiable",
                          iterations, nullptr);

  if (resultOut)
    *resultOut = MOORE_CONSTRAINT_ITERATION_LIMIT;

  // Return an unconstrained random value within the range
  return __moore_randomize_with_range(min, max);
}

extern "C" int64_t __moore_randomize_with_ranges_constrained(
    int64_t *ranges, int64_t numRanges, MooreConstraintPredicate predicate,
    void *userData, int64_t iterationLimit, int32_t *resultOut) {
  // Update statistics
  constraintState.totalAttempts.fetch_add(1);

  // Validate inputs
  if (!ranges || numRanges <= 0) {
    constraintState.fallbackCount.fetch_add(1);
    localStats.lastIterations = 0;
    if (resultOut)
      *resultOut = MOORE_CONSTRAINT_FALLBACK;
    return 0;
  }

  // Determine effective iteration limit
  int64_t limit = iterationLimit > 0 ? iterationLimit
                                     : constraintState.iterationLimit.load();

  // If no predicate, just do simple multi-range randomization
  if (!predicate) {
    constraintState.successfulSolves.fetch_add(1);
    localStats.lastIterations = 1;
    if (resultOut)
      *resultOut = MOORE_CONSTRAINT_SUCCESS;
    return __moore_randomize_with_ranges(ranges, numRanges);
  }

  // Try to find a value that satisfies the predicate
  int64_t iterations = 0;

  for (iterations = 0; iterations < limit; ++iterations) {
    // Generate random value from ranges
    int64_t value = __moore_randomize_with_ranges(ranges, numRanges);

    // Check if it satisfies the predicate
    if (predicate(value, userData)) {
      constraintState.successfulSolves.fetch_add(1);
      localStats.lastIterations = iterations + 1;
      if (resultOut)
        *resultOut = MOORE_CONSTRAINT_SUCCESS;
      return value;
    }
  }

  // Hit iteration limit - use fallback
  constraintState.iterationLimitHits.fetch_add(1);
  constraintState.fallbackCount.fetch_add(1);
  localStats.lastIterations = iterations;

  // Emit warning
  __moore_constraint_warn("Hit iteration limit on multi-range constraint",
                          iterations, nullptr);

  if (resultOut)
    *resultOut = MOORE_CONSTRAINT_ITERATION_LIMIT;

  // Return an unconstrained random value from the ranges (ignore predicate)
  return __moore_randomize_with_ranges(ranges, numRanges);
}

//===----------------------------------------------------------------------===//
// Pre/Post Randomize Callbacks
//===----------------------------------------------------------------------===//
//
// SystemVerilog supports pre_randomize() and post_randomize() callback methods
// that are invoked before and after randomization respectively.
// IEEE 1800-2017 Section 18.6.1 "Pre and post randomize methods".
//
// These are virtual methods that user classes can override to perform setup
// or post-processing around randomization. The default implementations are
// empty (no-ops).
//

/// Call pre_randomize() callback on a class object.
/// This is called before the randomization process begins.
///
/// NOTE: As of the current implementation, pre_randomize callbacks are handled
/// directly in the MooreToCore lowering pass by generating direct calls to the
/// user-defined pre_randomize method (e.g., "ClassName::pre_randomize"). This
/// runtime function is kept as a fallback stub for potential future vtable-
/// based dispatch or for compatibility with older generated code.
extern "C" void __moore_call_pre_randomize(void *classPtr) {
  // The direct call approach in MooreToCore handles this - this stub is for
  // backward compatibility and potential future vtable dispatch.
  (void)classPtr;
}

/// Call post_randomize() callback on a class object.
/// This is called after randomization succeeds.
///
/// NOTE: As of the current implementation, post_randomize callbacks are handled
/// directly in the MooreToCore lowering pass by generating direct calls to the
/// user-defined post_randomize method (e.g., "ClassName::post_randomize"). This
/// runtime function is kept as a fallback stub for potential future vtable-
/// based dispatch or for compatibility with older generated code.
extern "C" void __moore_call_post_randomize(void *classPtr) {
  // The direct call approach in MooreToCore handles this - this stub is for
  // backward compatibility and potential future vtable dispatch.
  (void)classPtr;
}

//===----------------------------------------------------------------------===//
// Constraint Mode Control
//===----------------------------------------------------------------------===//
//
// SystemVerilog supports constraint_mode() to enable/disable constraints.
// IEEE 1800-2017 Section 18.8 "Disabling random variables and constraints".
//
// constraint_mode(0) disables a constraint
// constraint_mode(1) enables a constraint
// constraint_mode() returns the current mode (0 or 1)
//

namespace {

/// Global constraint mode state: tracks enabled/disabled state per constraint.
/// Key: unique constraint identifier (object address + constraint name hash)
/// Value: enabled (1) or disabled (0)
std::unordered_map<uint64_t, int32_t> constraintModeState;
std::mutex constraintModeMutex;

/// Generate a unique key for a constraint instance.
uint64_t makeConstraintKey(void *classPtr, const char *constraintName) {
  uint64_t key = reinterpret_cast<uint64_t>(classPtr);
  if (constraintName) {
    // Simple string hash
    const char *p = constraintName;
    while (*p) {
      key = key * 31 + static_cast<uint64_t>(*p);
      ++p;
    }
  }
  return key;
}

} // anonymous namespace

/// Get the current constraint mode (1 = enabled, 0 = disabled).
/// Returns 1 if the constraint has not been explicitly disabled.
extern "C" int32_t __moore_constraint_mode_get(void *classPtr,
                                               const char *constraintName) {
  uint64_t key = makeConstraintKey(classPtr, constraintName);
  std::lock_guard<std::mutex> lock(constraintModeMutex);
  auto it = constraintModeState.find(key);
  if (it == constraintModeState.end()) {
    // Not found - default is enabled
    return 1;
  }
  return it->second;
}

/// Set the constraint mode and return the previous mode.
/// mode = 0: disable, mode = 1: enable
extern "C" int32_t __moore_constraint_mode_set(void *classPtr,
                                               const char *constraintName,
                                               int32_t mode) {
  uint64_t key = makeConstraintKey(classPtr, constraintName);
  std::lock_guard<std::mutex> lock(constraintModeMutex);
  auto it = constraintModeState.find(key);
  int32_t previousMode = (it == constraintModeState.end()) ? 1 : it->second;
  constraintModeState[key] = (mode != 0) ? 1 : 0;
  return previousMode;
}

/// Disable all constraints on a class object.
/// Returns 1 if any constraints were enabled, 0 otherwise.
extern "C" int32_t __moore_constraint_mode_disable_all(void *classPtr) {
  // This is a simplified implementation that sets a "disable all" flag.
  // In practice, we'd iterate through all constraint blocks for this class.
  uint64_t key = makeConstraintKey(classPtr, "__all__");
  std::lock_guard<std::mutex> lock(constraintModeMutex);
  auto it = constraintModeState.find(key);
  int32_t previousMode = (it == constraintModeState.end()) ? 1 : it->second;
  constraintModeState[key] = 0;
  return previousMode;
}

/// Enable all constraints on a class object.
/// Returns 1 if any constraints were disabled, 0 otherwise.
extern "C" int32_t __moore_constraint_mode_enable_all(void *classPtr) {
  uint64_t key = makeConstraintKey(classPtr, "__all__");
  std::lock_guard<std::mutex> lock(constraintModeMutex);
  auto it = constraintModeState.find(key);
  int32_t previousMode = (it == constraintModeState.end()) ? 1 : it->second;
  constraintModeState[key] = 1;
  return previousMode;
}

/// Check if a specific constraint is enabled.
/// Takes into account both individual constraint mode and "disable all" flag.
extern "C" int32_t __moore_is_constraint_enabled(void *classPtr,
                                                 const char *constraintName) {
  std::lock_guard<std::mutex> lock(constraintModeMutex);

  // First check the "disable all" flag
  uint64_t allKey = makeConstraintKey(classPtr, "__all__");
  auto allIt = constraintModeState.find(allKey);
  if (allIt != constraintModeState.end() && allIt->second == 0) {
    // All constraints disabled
    return 0;
  }

  // Then check individual constraint mode
  uint64_t key = makeConstraintKey(classPtr, constraintName);
  auto it = constraintModeState.find(key);
  if (it == constraintModeState.end()) {
    // Not found - default is enabled
    return 1;
  }
  return it->second;
}

//===----------------------------------------------------------------------===//
// Array Constraint Operations
//===----------------------------------------------------------------------===//
//
// These functions provide runtime support for array constraint features:
// - Unique constraints: ensure all elements have distinct values
// - Foreach constraints: element-wise constraint validation
// - Size constraints: array size validation
// - Sum constraints: aggregate constraint validation
//
// IEEE 1800-2017 Section 18.5.5 "Uniqueness constraints"
// IEEE 1800-2017 Section 18.5.8 "Foreach constraints"
//

namespace {

/// Helper to read an element value as uint64_t from memory.
uint64_t readElementAsUint64(void *ptr, int64_t elementSize) {
  uint64_t value = 0;
  if (elementSize > 8)
    elementSize = 8;
  std::memcpy(&value, ptr, static_cast<size_t>(elementSize));
  return value;
}

/// Helper to write a uint64_t value to memory as an element.
void writeElementFromUint64(void *ptr, int64_t elementSize, uint64_t value) {
  if (elementSize > 8)
    elementSize = 8;
  std::memcpy(ptr, &value, static_cast<size_t>(elementSize));
}

//===----------------------------------------------------------------------===//
// Implication Constraint Operations
//===----------------------------------------------------------------------===//
//
// These functions provide runtime support for implication constraints:
// - Basic implication: antecedent -> consequent
// - Nested implication checking
// - Implication with soft/hard constraint handling
//
// IEEE 1800-2017 Section 18.5.6 "Implication constraints"
// IEEE 1800-2017 Section 18.5.7 "if-else constraints"
//

/// Global statistics for implication constraint evaluation.
MooreImplicationStats gImplicationStats = {0, 0, 0, 0};

} // anonymous namespace

/// Check if an implication constraint is satisfied.
/// Implements the SystemVerilog implication operator: antecedent -> consequent
/// If antecedent is false (0), the implication is trivially true.
/// If antecedent is true (non-zero), the consequent must be true (non-zero).
extern "C" int32_t __moore_constraint_check_implication(int32_t antecedent,
                                                         int32_t consequent) {
  // Update statistics
  gImplicationStats.totalImplications++;

  // Implication truth table:
  // antecedent | consequent | result
  //     0      |     0      |   1 (vacuously true)
  //     0      |     1      |   1 (vacuously true)
  //     1      |     0      |   0 (constraint violated)
  //     1      |     1      |   1 (constraint satisfied)

  if (antecedent != 0) {
    gImplicationStats.triggeredImplications++;
    if (consequent != 0) {
      gImplicationStats.satisfiedImplications++;
      return 1;
    }
    return 0; // Constraint violated: antecedent true but consequent false
  }

  // Antecedent is false, implication is vacuously true
  gImplicationStats.satisfiedImplications++;
  return 1;
}

/// Check a nested implication constraint (a -> (b -> c)).
/// Evaluates as: !a || (!b || c) which is equivalent to: !a || !b || c
extern "C" int32_t __moore_constraint_check_nested_implication(int32_t outer,
                                                                int32_t inner,
                                                                int32_t consequent) {
  // Nested implication: outer -> (inner -> consequent)
  // Equivalent to: !outer || !inner || consequent
  // Or as implications: if outer is true, check inner -> consequent

  gImplicationStats.totalImplications++;

  if (outer == 0) {
    // Outer antecedent is false, entire implication is vacuously true
    gImplicationStats.satisfiedImplications++;
    return 1;
  }

  gImplicationStats.triggeredImplications++;

  if (inner == 0) {
    // Inner antecedent is false, inner implication is true, so whole is true
    gImplicationStats.satisfiedImplications++;
    return 1;
  }

  // Both antecedents are true, consequent must be true
  if (consequent != 0) {
    gImplicationStats.satisfiedImplications++;
    return 1;
  }

  return 0; // Constraint violated
}

/// Evaluate an implication constraint and apply soft/hard semantics.
/// Soft implications provide a preference but don't cause constraint failure.
/// Hard implications are enforced strictly and cause failure if violated.
extern "C" int32_t __moore_constraint_check_implication_soft(int32_t antecedent,
                                                              int32_t consequentSatisfied,
                                                              int32_t isSoft) {
  gImplicationStats.totalImplications++;

  if (antecedent == 0) {
    // Antecedent is false, implication is vacuously satisfied
    gImplicationStats.satisfiedImplications++;
    return 1;
  }

  gImplicationStats.triggeredImplications++;

  if (consequentSatisfied != 0) {
    // Consequent is satisfied
    gImplicationStats.satisfiedImplications++;
    return 1;
  }

  // Consequent not satisfied
  if (isSoft != 0) {
    // Soft constraint - use fallback behavior, don't fail
    gImplicationStats.softFallbacks++;
    gImplicationStats.satisfiedImplications++;
    return 1;
  }

  // Hard constraint violated
  return 0;
}

/// Get global implication constraint statistics.
extern "C" MooreImplicationStats *__moore_implication_get_stats(void) {
  return &gImplicationStats;
}

/// Reset global implication statistics to zero.
extern "C" void __moore_implication_reset_stats(void) {
  gImplicationStats.totalImplications = 0;
  gImplicationStats.triggeredImplications = 0;
  gImplicationStats.satisfiedImplications = 0;
  gImplicationStats.softFallbacks = 0;
}

//===----------------------------------------------------------------------===//
// Array Constraint Operations
//===----------------------------------------------------------------------===//

/// Check if all elements in an array are unique.
/// This implements the SystemVerilog `unique {arr}` constraint.
extern "C" int32_t __moore_constraint_unique_check(void *array,
                                                   int64_t numElements,
                                                   int64_t elementSize) {
  if (!array || numElements <= 1 || elementSize <= 0)
    return 1; // Empty or single-element arrays are trivially unique

  auto *data = static_cast<char *>(array);

  // Use a set to track seen values for efficiency
  std::set<std::vector<uint8_t>> seenValues;

  for (int64_t i = 0; i < numElements; ++i) {
    std::vector<uint8_t> elemBytes(static_cast<size_t>(elementSize));
    std::memcpy(elemBytes.data(), data + i * elementSize,
                static_cast<size_t>(elementSize));

    if (!seenValues.insert(elemBytes).second) {
      // Duplicate found
      return 0;
    }
  }

  return 1; // All elements are unique
}

/// Check if multiple scalar variables are all unique.
/// This implements the SystemVerilog `unique {a, b, c}` constraint.
extern "C" int32_t __moore_constraint_unique_scalars(void *values,
                                                     int64_t numValues,
                                                     int64_t valueSize) {
  // Same implementation as array check
  return __moore_constraint_unique_check(values, numValues, valueSize);
}

/// Randomize an array ensuring all elements are unique.
/// Uses Fisher-Yates shuffle on a range of values.
extern "C" int32_t __moore_randomize_unique_array(void *array,
                                                  int64_t numElements,
                                                  int64_t elementSize,
                                                  int64_t minValue,
                                                  int64_t maxValue) {
  if (!array || numElements <= 0 || elementSize <= 0)
    return 0;

  // Check if range is large enough
  int64_t rangeSize = maxValue - minValue + 1;
  if (rangeSize < numElements) {
    // Cannot generate enough unique values
    return 0;
  }

  auto *data = static_cast<char *>(array);

  // For small ranges, use Fisher-Yates on the full range
  if (rangeSize <= 100000 && rangeSize <= numElements * 10) {
    std::vector<int64_t> pool;
    pool.reserve(static_cast<size_t>(rangeSize));
    for (int64_t v = minValue; v <= maxValue; ++v) {
      pool.push_back(v);
    }

    // Shuffle and pick first numElements
    for (int64_t i = 0; i < numElements; ++i) {
      int64_t remaining = static_cast<int64_t>(pool.size()) - i;
      int64_t j = i + (std::rand() % remaining);
      std::swap(pool[static_cast<size_t>(i)], pool[static_cast<size_t>(j)]);

      // Write to array
      writeElementFromUint64(data + i * elementSize, elementSize,
                             static_cast<uint64_t>(pool[static_cast<size_t>(i)]));
    }
    return 1;
  }

  // For large ranges, use rejection sampling
  std::set<int64_t> usedValues;
  int64_t maxIterations = numElements * 100; // Prevent infinite loops
  int64_t iterations = 0;

  for (int64_t i = 0; i < numElements; ++i) {
    int64_t value;
    do {
      value = minValue + (std::rand() % rangeSize);
      iterations++;
      if (iterations > maxIterations) {
        // Fallback: just fill with sequential values
        for (int64_t j = i; j < numElements; ++j) {
          writeElementFromUint64(data + j * elementSize, elementSize,
                                 static_cast<uint64_t>(minValue + j));
        }
        return 1;
      }
    } while (usedValues.count(value) > 0);

    usedValues.insert(value);
    writeElementFromUint64(data + i * elementSize, elementSize,
                           static_cast<uint64_t>(value));
  }

  return 1;
}

/// Validate a foreach constraint on an array.
/// Checks that all elements satisfy a predicate function.
extern "C" int32_t __moore_constraint_foreach_validate(
    void *array, int64_t numElements, int64_t elementSize,
    bool (*predicate)(int64_t, void *), void *userData) {
  if (!array || numElements <= 0 || elementSize <= 0 || !predicate)
    return 1; // Empty arrays or no predicate trivially satisfy

  auto *data = static_cast<char *>(array);

  for (int64_t i = 0; i < numElements; ++i) {
    int64_t value = static_cast<int64_t>(
        readElementAsUint64(data + i * elementSize, elementSize));
    if (!predicate(value, userData)) {
      return 0; // Constraint violated
    }
  }

  return 1; // All elements satisfy the constraint
}

/// Validate an array size constraint.
/// Checks that the array has exactly the expected number of elements.
extern "C" int32_t __moore_constraint_size_check(MooreQueue *array,
                                                 int64_t expectedSize) {
  if (!array)
    return expectedSize == 0 ? 1 : 0;

  return array->len == expectedSize ? 1 : 0;
}

/// Validate an array sum constraint.
/// Checks that the sum of all elements equals the expected value.
extern "C" int32_t __moore_constraint_sum_check(MooreQueue *array,
                                                int64_t elementSize,
                                                int64_t expectedSum) {
  if (!array || !array->data || array->len <= 0)
    return expectedSum == 0 ? 1 : 0;

  int64_t actualSum = __moore_array_reduce_sum(array, elementSize);
  return actualSum == expectedSum ? 1 : 0;
}

//===----------------------------------------------------------------------===//
// Rand Mode Control
//===----------------------------------------------------------------------===//
//
// SystemVerilog supports rand_mode() to enable/disable random variables.
// IEEE 1800-2017 Section 18.8 "Disabling random variables and constraints".
//
// rand_mode(0) disables a random variable
// rand_mode(1) enables a random variable
// rand_mode() returns the current mode (0 or 1)
//

namespace {

/// Global rand mode state: tracks enabled/disabled state per random variable.
/// Key: unique identifier (object address + property name hash)
/// Value: enabled (1) or disabled (0)
std::unordered_map<uint64_t, int32_t> randModeState;
std::mutex randModeMutex;

} // anonymous namespace

/// Get the current rand mode (1 = enabled, 0 = disabled).
/// Returns 1 if the variable has not been explicitly disabled.
extern "C" int32_t __moore_rand_mode_get(void *classPtr,
                                         const char *propertyName) {
  uint64_t key = makeConstraintKey(classPtr, propertyName);
  std::lock_guard<std::mutex> lock(randModeMutex);
  auto it = randModeState.find(key);
  if (it == randModeState.end()) {
    // Not found - default is enabled
    return 1;
  }
  return it->second;
}

/// Set the rand mode and return the previous mode.
/// mode = 0: disable, mode = 1: enable
extern "C" int32_t __moore_rand_mode_set(void *classPtr,
                                         const char *propertyName,
                                         int32_t mode) {
  uint64_t key = makeConstraintKey(classPtr, propertyName);
  std::lock_guard<std::mutex> lock(randModeMutex);
  auto it = randModeState.find(key);
  int32_t previousMode = (it == randModeState.end()) ? 1 : it->second;
  randModeState[key] = (mode != 0) ? 1 : 0;
  return previousMode;
}

/// Disable all random variables on a class object.
/// Returns 1 if any variables were enabled, 0 otherwise.
extern "C" int32_t __moore_rand_mode_disable_all(void *classPtr) {
  uint64_t key = makeConstraintKey(classPtr, "__all__");
  std::lock_guard<std::mutex> lock(randModeMutex);
  auto it = randModeState.find(key);
  int32_t previousMode = (it == randModeState.end()) ? 1 : it->second;
  randModeState[key] = 0;
  return previousMode;
}

/// Enable all random variables on a class object.
/// Returns 1 if any variables were disabled, 0 otherwise.
extern "C" int32_t __moore_rand_mode_enable_all(void *classPtr) {
  uint64_t key = makeConstraintKey(classPtr, "__all__");
  std::lock_guard<std::mutex> lock(randModeMutex);
  auto it = randModeState.find(key);
  int32_t previousMode = (it == randModeState.end()) ? 1 : it->second;
  randModeState[key] = 1;
  return previousMode;
}

/// Check if a specific random variable is enabled.
/// Takes into account both individual rand mode and "disable all" flag.
extern "C" int32_t __moore_is_rand_enabled(void *classPtr,
                                           const char *propertyName) {
  std::lock_guard<std::mutex> lock(randModeMutex);

  // First check the "disable all" flag
  uint64_t allKey = makeConstraintKey(classPtr, "__all__");
  auto allIt = randModeState.find(allKey);
  if (allIt != randModeState.end() && allIt->second == 0) {
    // All random variables disabled
    return 0;
  }

  // Then check individual rand mode
  uint64_t key = makeConstraintKey(classPtr, propertyName);
  auto it = randModeState.find(key);
  if (it == randModeState.end()) {
    // Not found - default is enabled
    return 1;
  }
  return it->second;
}

//===----------------------------------------------------------------------===//
// File I/O Operations
//===----------------------------------------------------------------------===//
//
// SystemVerilog file I/O uses multichannel descriptors (MCDs) where the
// low 32 bits represent different output channels. Bit 0 is stdout,
// and bits 1-30 are file channels. We implement a simplified version
// that maps MCDs directly to FILE* handles.
//

namespace {

/// Maximum number of open files (excluding stdout/stderr).
/// SystemVerilog supports up to 31 file channels (bits 1-30 of MCD).
constexpr int32_t kMaxOpenFiles = 31;

/// File handle table: maps file descriptor bits to FILE* handles.
/// Index 0 is reserved (stdout), indices 1-30 are file channels.
thread_local FILE *fileHandles[kMaxOpenFiles] = {nullptr};

/// Find the first available file descriptor slot.
/// Returns the slot index (1-30), or -1 if no slots available.
int32_t findFreeSlot() {
  for (int32_t i = 1; i < kMaxOpenFiles; ++i) {
    if (fileHandles[i] == nullptr)
      return i;
  }
  return -1;
}

/// Convert MooreString to null-terminated C string.
/// Caller is responsible for freeing the returned string.
char *toCString(MooreString *str) {
  if (!str || !str->data || str->len <= 0)
    return nullptr;
  char *cstr = static_cast<char *>(std::malloc(str->len + 1));
  if (!cstr)
    return nullptr;
  std::memcpy(cstr, str->data, str->len);
  cstr[str->len] = '\0';
  return cstr;
}

} // anonymous namespace

extern "C" int32_t __moore_fopen(MooreString *filename, MooreString *mode) {
  // Validate filename
  if (!filename || !filename->data || filename->len <= 0)
    return 0;

  // Find available file descriptor slot
  int32_t slot = findFreeSlot();
  if (slot < 0)
    return 0; // No available slots

  // Convert filename to C string
  char *fnameStr = toCString(filename);
  if (!fnameStr)
    return 0;

  // Determine file mode (default to "r" if not specified)
  const char *modeStr = "r";
  char *allocatedMode = nullptr;
  if (mode && mode->data && mode->len > 0) {
    allocatedMode = toCString(mode);
    if (allocatedMode)
      modeStr = allocatedMode;
  }

  // Open the file
  FILE *fp = std::fopen(fnameStr, modeStr);

  // Clean up allocated strings
  std::free(fnameStr);
  if (allocatedMode)
    std::free(allocatedMode);

  if (!fp)
    return 0;

  // Store the file handle and return the MCD
  // The MCD for file channel i is (1 << i)
  fileHandles[slot] = fp;
  return 1 << slot;
}

extern "C" void __moore_fwrite(int32_t fd, MooreString *message) {
  // Validate message
  if (!message || !message->data || message->len <= 0)
    return;

  // Handle stdout (bit 0 of MCD)
  if (fd & 1) {
    std::fwrite(message->data, 1, message->len, stdout);
  }

  // Handle file channels (bits 1-30)
  for (int32_t i = 1; i < kMaxOpenFiles; ++i) {
    if ((fd & (1 << i)) && fileHandles[i]) {
      std::fwrite(message->data, 1, message->len, fileHandles[i]);
    }
  }
}

extern "C" void __moore_fclose(int32_t fd) {
  // Close all file channels indicated by the MCD
  // Note: We don't close stdout (bit 0)
  for (int32_t i = 1; i < kMaxOpenFiles; ++i) {
    if ((fd & (1 << i)) && fileHandles[i]) {
      std::fclose(fileHandles[i]);
      fileHandles[i] = nullptr;
    }
  }
}

extern "C" int32_t __moore_fgetc(int32_t fd) {
  // Find the first file indicated by the MCD
  for (int32_t i = 1; i < kMaxOpenFiles; ++i) {
    if ((fd & (1 << i)) && fileHandles[i]) {
      return std::fgetc(fileHandles[i]);
    }
  }
  return -1; // EOF/error
}

extern "C" int32_t __moore_fgets(MooreString *str, int32_t fd) {
  if (!str)
    return 0;

  // Find the first file indicated by the MCD
  FILE *file = nullptr;
  for (int32_t i = 1; i < kMaxOpenFiles; ++i) {
    if ((fd & (1 << i)) && fileHandles[i]) {
      file = fileHandles[i];
      break;
    }
  }

  if (!file)
    return 0;

  // Read line into a temporary buffer
  char buffer[4096];
  if (!std::fgets(buffer, sizeof(buffer), file))
    return 0;

  // Allocate string and copy data
  size_t len = std::strlen(buffer);
  str->data = static_cast<char *>(std::malloc(len + 1));
  if (!str->data) {
    str->len = 0;
    return 0;
  }
  std::memcpy(str->data, buffer, len + 1);
  str->len = static_cast<int64_t>(len);

  return static_cast<int32_t>(len);
}

extern "C" int32_t __moore_feof(int32_t fd) {
  // Find the first file indicated by the MCD
  for (int32_t i = 1; i < kMaxOpenFiles; ++i) {
    if ((fd & (1 << i)) && fileHandles[i]) {
      return std::feof(fileHandles[i]) ? 1 : 0;
    }
  }
  return 1; // Treat invalid fd as EOF
}

extern "C" void __moore_fflush(int32_t fd) {
  if (fd == 0) {
    // Flush all open files
    for (int32_t i = 1; i < kMaxOpenFiles; ++i) {
      if (fileHandles[i]) {
        std::fflush(fileHandles[i]);
      }
    }
    std::fflush(stdout);
  } else {
    // Flush specific files indicated by the MCD
    for (int32_t i = 1; i < kMaxOpenFiles; ++i) {
      if ((fd & (1 << i)) && fileHandles[i]) {
        std::fflush(fileHandles[i]);
      }
    }
  }
}

extern "C" int32_t __moore_ftell(int32_t fd) {
  // Find the first file indicated by the MCD
  for (int32_t i = 1; i < kMaxOpenFiles; ++i) {
    if ((fd & (1 << i)) && fileHandles[i]) {
      return static_cast<int32_t>(std::ftell(fileHandles[i]));
    }
  }
  return -1; // Error
}

//===----------------------------------------------------------------------===//
// Display System Tasks
//===----------------------------------------------------------------------===//
//
// Implementation of SystemVerilog display system tasks: $display, $write,
// $strobe, and $monitor. These functions output pre-formatted messages
// that have been processed by the Moore dialect's format string operations.
//

namespace {
// Simulation time tracking for $time system function
static int64_t currentSimulationTime = 0;
static std::mutex simTimeMutex;

// Strobe queue for end-of-timestep output
struct StrobeEntry {
  std::string message;
};
static std::vector<StrobeEntry> strobeQueue;
static std::mutex strobeMutex;

// Monitor state for $monitor system task
struct MonitorState {
  std::string messageTemplate;
  std::vector<void *> valuePointers;
  std::vector<int32_t> valueSizes;
  std::vector<std::vector<uint8_t>> lastValues;
  bool enabled = true;
  bool active = false;
};
static MonitorState monitorState;
static std::mutex monitorMutex;

// Helper to copy a monitored value for comparison
static std::vector<uint8_t> copyValue(void *ptr, int32_t size) {
  std::vector<uint8_t> result(static_cast<size_t>(size));
  if (ptr && size > 0) {
    std::memcpy(result.data(), ptr, static_cast<size_t>(size));
  }
  return result;
}

// Helper to check if a value has changed
static bool valueChanged(void *ptr, int32_t size,
                         const std::vector<uint8_t> &lastValue) {
  if (!ptr || size <= 0 || static_cast<size_t>(size) != lastValue.size())
    return true;
  return std::memcmp(ptr, lastValue.data(), static_cast<size_t>(size)) != 0;
}
} // namespace

extern "C" void __moore_display(MooreString *message) {
  // Validate message
  if (!message)
    return;

  // Print the message content
  if (message->data && message->len > 0) {
    std::fwrite(message->data, 1, static_cast<size_t>(message->len), stdout);
  }

  // $display always appends a newline
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

extern "C" void __moore_write(MooreString *message) {
  // Validate message
  if (!message)
    return;

  // Print the message content without newline
  if (message->data && message->len > 0) {
    std::fwrite(message->data, 1, static_cast<size_t>(message->len), stdout);
    std::fflush(stdout);
  }
}

extern "C" void __moore_strobe(MooreString *message) {
  // Validate message
  if (!message || !message->data || message->len <= 0)
    return;

  // Queue the message for end-of-timestep output
  std::lock_guard<std::mutex> lock(strobeMutex);
  StrobeEntry entry;
  entry.message.assign(message->data, static_cast<size_t>(message->len));
  strobeQueue.push_back(std::move(entry));
}

extern "C" void __moore_monitor(MooreString *message, void **values,
                                int32_t numValues, int32_t *valueSizes) {
  std::lock_guard<std::mutex> lock(monitorMutex);

  // Clear any previous monitor state
  monitorState.valuePointers.clear();
  monitorState.valueSizes.clear();
  monitorState.lastValues.clear();
  monitorState.active = false;

  // Validate inputs
  if (!message || !message->data || message->len <= 0) {
    return;
  }

  // Store the message template
  monitorState.messageTemplate.assign(message->data,
                                      static_cast<size_t>(message->len));

  // Store value pointers and sizes
  if (values && valueSizes && numValues > 0) {
    for (int32_t i = 0; i < numValues; ++i) {
      monitorState.valuePointers.push_back(values[i]);
      monitorState.valueSizes.push_back(valueSizes[i]);
      // Initialize last values for change detection
      monitorState.lastValues.push_back(copyValue(values[i], valueSizes[i]));
    }
  }

  monitorState.active = true;

  // Display initial values (monitor triggers immediately on setup)
  if (monitorState.enabled) {
    std::fputs(monitorState.messageTemplate.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
  }
}

extern "C" void __moore_monitoroff(void) {
  std::lock_guard<std::mutex> lock(monitorMutex);
  monitorState.enabled = false;
}

extern "C" void __moore_monitoron(void) {
  std::lock_guard<std::mutex> lock(monitorMutex);
  monitorState.enabled = true;
}

extern "C" void __moore_print_dyn_string(MooreString *str) {
  // Simply print the dynamic string content without newline
  if (str && str->data && str->len > 0) {
    std::fwrite(str->data, 1, static_cast<size_t>(str->len), stdout);
  }
}

extern "C" int64_t __moore_get_time(void) {
  std::lock_guard<std::mutex> lock(simTimeMutex);
  return currentSimulationTime;
}

extern "C" void __moore_set_time(int64_t time) {
  std::lock_guard<std::mutex> lock(simTimeMutex);
  currentSimulationTime = time;
}

extern "C" void __moore_strobe_flush(void) {
  std::lock_guard<std::mutex> lock(strobeMutex);

  // Output all queued strobe messages
  for (const auto &entry : strobeQueue) {
    std::fputs(entry.message.c_str(), stdout);
    std::fputc('\n', stdout);
  }

  // Clear the queue
  strobeQueue.clear();

  std::fflush(stdout);
}

extern "C" void __moore_monitor_check(void) {
  std::lock_guard<std::mutex> lock(monitorMutex);

  // Skip if monitor is not active or disabled
  if (!monitorState.active || !monitorState.enabled)
    return;

  // Check if any monitored value has changed
  bool anyChanged = false;
  for (size_t i = 0; i < monitorState.valuePointers.size(); ++i) {
    void *ptr = monitorState.valuePointers[i];
    int32_t size = monitorState.valueSizes[i];
    if (valueChanged(ptr, size, monitorState.lastValues[i])) {
      anyChanged = true;
      // Update the last value
      monitorState.lastValues[i] = copyValue(ptr, size);
    }
  }

  // If any value changed, display the monitor message
  if (anyChanged) {
    std::fputs(monitorState.messageTemplate.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
  }
}

//===----------------------------------------------------------------------===//
// Simulation Control Tasks
//===----------------------------------------------------------------------===//

namespace {
// Global state for simulation control and severity tracking
struct SimulationControlState {
  std::atomic<int32_t> errorCount{0};
  std::atomic<int32_t> warningCount{0};
  std::atomic<bool> finishCalled{false};
  std::atomic<int32_t> exitCode{0};
  std::atomic<bool> finishExits{true}; // Default: actually exit on $finish
};

SimulationControlState simControlState;
std::mutex simControlMutex;
} // namespace

extern "C" void __moore_finish(int32_t exit_code) {
  std::lock_guard<std::mutex> lock(simControlMutex);
  simControlState.finishCalled.store(true);
  simControlState.exitCode.store(exit_code);

  // Print finish message to stderr
  std::fprintf(stderr, "$finish called with exit code %d\n", exit_code);
  std::fflush(stderr);

  // If configured to actually exit, do so
  if (simControlState.finishExits.load()) {
    std::exit(exit_code);
  }
}

extern "C" void __moore_fatal(int32_t exit_code, MooreString *message) {
  std::lock_guard<std::mutex> lock(simControlMutex);

  // Print fatal message to stderr
  std::fprintf(stderr, "Fatal: ");
  if (message && message->data && message->len > 0) {
    std::fwrite(message->data, 1, static_cast<size_t>(message->len), stderr);
  }
  std::fputc('\n', stderr);
  std::fflush(stderr);

  // Increment error count (fatal is an error)
  simControlState.errorCount.fetch_add(1);
  simControlState.finishCalled.store(true);
  simControlState.exitCode.store(exit_code);

  // If configured to actually exit, do so
  if (simControlState.finishExits.load()) {
    std::exit(exit_code);
  }
}

extern "C" void __moore_error(MooreString *message) {
  // Print error message to stderr
  std::fprintf(stderr, "Error: ");
  if (message && message->data && message->len > 0) {
    std::fwrite(message->data, 1, static_cast<size_t>(message->len), stderr);
  }
  std::fputc('\n', stderr);
  std::fflush(stderr);

  // Increment error count
  simControlState.errorCount.fetch_add(1);
}

extern "C" void __moore_warning(MooreString *message) {
  // Print warning message to stderr
  std::fprintf(stderr, "Warning: ");
  if (message && message->data && message->len > 0) {
    std::fwrite(message->data, 1, static_cast<size_t>(message->len), stderr);
  }
  std::fputc('\n', stderr);
  std::fflush(stderr);

  // Increment warning count
  simControlState.warningCount.fetch_add(1);
}

extern "C" void __moore_info(MooreString *message) {
  // Print info message to stdout (info is not an error)
  std::fprintf(stdout, "Info: ");
  if (message && message->data && message->len > 0) {
    std::fwrite(message->data, 1, static_cast<size_t>(message->len), stdout);
  }
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

extern "C" int32_t __moore_get_error_count(void) {
  return simControlState.errorCount.load();
}

extern "C" int32_t __moore_get_warning_count(void) {
  return simControlState.warningCount.load();
}

extern "C" void __moore_reset_severity_counts(void) {
  simControlState.errorCount.store(0);
  simControlState.warningCount.store(0);
}

extern "C" int32_t __moore_severity_summary(void) {
  int32_t errors = simControlState.errorCount.load();
  int32_t warnings = simControlState.warningCount.load();

  if (errors > 0 || warnings > 0) {
    std::fprintf(stderr,
                 "Simulation completed with %d error(s) and %d warning(s)\n",
                 errors, warnings);
    std::fflush(stderr);
  }

  return errors;
}

extern "C" void __moore_set_finish_exits(bool should_exit) {
  simControlState.finishExits.store(should_exit);
}

extern "C" bool __moore_finish_called(void) {
  return simControlState.finishCalled.load();
}

extern "C" int32_t __moore_get_exit_code(void) {
  return simControlState.exitCode.load();
}

extern "C" void __moore_reset_finish_state(void) {
  std::lock_guard<std::mutex> lock(simControlMutex);
  simControlState.finishCalled.store(false);
  simControlState.exitCode.store(0);
  simControlState.errorCount.store(0);
  simControlState.warningCount.store(0);
}

//===----------------------------------------------------------------------===//
// Memory Management
//===----------------------------------------------------------------------===//

extern "C" void __moore_free(void *ptr) {
  std::free(ptr);
}

//===----------------------------------------------------------------------===//
// DPI-C Import Stubs for UVM Support
//===----------------------------------------------------------------------===//
//
// These stub implementations provide basic functionality for UVM DPI-C imports.
// They allow UVM code to compile and run without requiring external C libraries.
// For production use, these should be replaced with full implementations.
//

//===----------------------------------------------------------------------===//
// Signal Registry Bridge
//===----------------------------------------------------------------------===//

namespace {
/// Entry for a registered signal in the signal registry
struct SignalRegistryEntry {
  MooreSignalHandle handle;
  uint32_t width;
};

/// Global signal registry state
struct SignalRegistryState {
  /// Map from hierarchical path to signal entry
  std::unordered_map<std::string, SignalRegistryEntry> signals;

  /// Accessor callbacks
  MooreSignalReadCallback readCallback = nullptr;
  MooreSignalWriteCallback writeCallback = nullptr;
  MooreSignalForceCallback forceCallback = nullptr;
  MooreSignalReleaseCallback releaseCallback = nullptr;
  void *userData = nullptr;

  /// Check if the registry is connected to actual simulation
  bool isConnected() const { return readCallback != nullptr; }
};

SignalRegistryState signalRegistry;
std::mutex signalRegistryMutex;
} // namespace

extern "C" int32_t __moore_signal_registry_register(const char *path,
                                                     MooreSignalHandle signalHandle,
                                                     uint32_t width) {
  if (!path || !*path || signalHandle == MOORE_INVALID_SIGNAL_HANDLE)
    return 0;

  std::lock_guard<std::mutex> lock(signalRegistryMutex);
  signalRegistry.signals[std::string(path)] = {signalHandle, width};
  return 1;
}

extern "C" void __moore_signal_registry_set_accessor(
    MooreSignalReadCallback readCallback, MooreSignalWriteCallback writeCallback,
    MooreSignalForceCallback forceCallback,
    MooreSignalReleaseCallback releaseCallback, void *userData) {
  std::lock_guard<std::mutex> lock(signalRegistryMutex);
  signalRegistry.readCallback = readCallback;
  signalRegistry.writeCallback = writeCallback;
  signalRegistry.forceCallback = forceCallback;
  signalRegistry.releaseCallback = releaseCallback;
  signalRegistry.userData = userData;
}

extern "C" MooreSignalHandle __moore_signal_registry_lookup(const char *path) {
  if (!path || !*path)
    return MOORE_INVALID_SIGNAL_HANDLE;

  std::lock_guard<std::mutex> lock(signalRegistryMutex);
  auto it = signalRegistry.signals.find(std::string(path));
  if (it != signalRegistry.signals.end())
    return it->second.handle;
  return MOORE_INVALID_SIGNAL_HANDLE;
}

extern "C" int32_t __moore_signal_registry_exists(const char *path) {
  if (!path || !*path)
    return 0;

  std::lock_guard<std::mutex> lock(signalRegistryMutex);
  return signalRegistry.signals.count(std::string(path)) > 0 ? 1 : 0;
}

extern "C" uint32_t __moore_signal_registry_get_width(const char *path) {
  if (!path || !*path)
    return 0;

  std::lock_guard<std::mutex> lock(signalRegistryMutex);
  auto it = signalRegistry.signals.find(std::string(path));
  if (it != signalRegistry.signals.end())
    return it->second.width;
  return 0;
}

extern "C" void __moore_signal_registry_clear(void) {
  std::lock_guard<std::mutex> lock(signalRegistryMutex);
  signalRegistry.signals.clear();
  // Note: We don't clear the callbacks - they're managed by the simulation
}

extern "C" uint64_t __moore_signal_registry_count(void) {
  std::lock_guard<std::mutex> lock(signalRegistryMutex);
  return static_cast<uint64_t>(signalRegistry.signals.size());
}

extern "C" int32_t __moore_signal_registry_is_connected(void) {
  std::lock_guard<std::mutex> lock(signalRegistryMutex);
  return signalRegistry.isConnected() ? 1 : 0;
}

//===----------------------------------------------------------------------===//
// Signal Registry - Hierarchy Traversal
//===----------------------------------------------------------------------===//

namespace {
/// Tracks forced signal state separately from simulation values.
/// This is needed because DPI force/release operates at the HDL access layer.
struct ForcedSignalEntry {
  MooreSignalHandle handle;
  int64_t forcedValue;
  bool isForced;
};

/// Global forced signal tracking (path -> forced state).
std::unordered_map<std::string, ForcedSignalEntry> forcedSignals;
std::mutex forcedSignalsMutex;

/// Parse array indices from a path component like "mem[5]" or "arr[3][2]".
/// Returns the base name and fills indices vector.
/// Example: "mem[5]" -> returns "mem", indices = {5}
///          "arr[3][2]" -> returns "arr", indices = {3, 2}
std::string parseArrayIndices(const std::string &component,
                              std::vector<int64_t> &indices) {
  indices.clear();
  size_t bracketPos = component.find('[');
  if (bracketPos == std::string::npos) {
    return component;
  }

  std::string baseName = component.substr(0, bracketPos);
  size_t pos = bracketPos;

  while (pos < component.size() && component[pos] == '[') {
    size_t endBracket = component.find(']', pos);
    if (endBracket == std::string::npos) {
      break;
    }
    std::string indexStr = component.substr(pos + 1, endBracket - pos - 1);
    // Check if indexStr is a valid integer without using exceptions
    bool valid = !indexStr.empty();
    for (size_t i = 0; i < indexStr.size() && valid; ++i) {
      if (i == 0 && indexStr[i] == '-') continue;
      if (!std::isdigit(static_cast<unsigned char>(indexStr[i]))) valid = false;
    }
    if (!valid) {
      indices.clear();
      return baseName;
    }
    indices.push_back(std::stoll(indexStr));
    pos = endBracket + 1;
  }

  return baseName;
}

/// Parse a hierarchical path into components.
/// Handles both "." and "/" as separators.
/// Example: "top.inst1.inst2.sig" -> {"top", "inst1", "inst2", "sig"}
std::vector<std::string> parseHierarchicalPath(const std::string &path) {
  std::vector<std::string> components;
  std::string current;

  for (char c : path) {
    if (c == '.' || c == '/') {
      if (!current.empty()) {
        components.push_back(current);
        current.clear();
      }
    } else {
      current += c;
    }
  }

  if (!current.empty()) {
    components.push_back(current);
  }

  return components;
}

/// Try to find a signal by looking up various path formats.
/// Handles cases where the path might be relative or use different separators.
MooreSignalHandle lookupSignalWithAlternatives(const std::string &path) {
  // Direct lookup
  auto it = signalRegistry.signals.find(path);
  if (it != signalRegistry.signals.end()) {
    return it->second.handle;
  }

  // Try with dot-separated components
  std::vector<std::string> components = parseHierarchicalPath(path);
  if (components.empty()) {
    return MOORE_INVALID_SIGNAL_HANDLE;
  }

  // Try just the signal name (last component)
  const std::string &signalName = components.back();
  it = signalRegistry.signals.find(signalName);
  if (it != signalRegistry.signals.end()) {
    return it->second.handle;
  }

  // Try partial paths from the end
  for (size_t i = components.size() - 1; i > 0; --i) {
    std::string partialPath;
    for (size_t j = i; j < components.size(); ++j) {
      if (!partialPath.empty()) {
        partialPath += ".";
      }
      partialPath += components[j];
    }
    it = signalRegistry.signals.find(partialPath);
    if (it != signalRegistry.signals.end()) {
      return it->second.handle;
    }
  }

  // Handle array indices - try base name without indices
  std::vector<int64_t> indices;
  std::string baseName = parseArrayIndices(signalName, indices);
  if (!indices.empty()) {
    it = signalRegistry.signals.find(baseName);
    if (it != signalRegistry.signals.end()) {
      // TODO: Handle array element access via index calculation
      return it->second.handle;
    }
  }

  return MOORE_INVALID_SIGNAL_HANDLE;
}

/// Get all signals that match a glob pattern.
/// Patterns can use * for wildcards.
std::vector<std::pair<std::string, MooreSignalHandle>>
matchSignalsByPattern(const std::string &pattern) {
  std::vector<std::pair<std::string, MooreSignalHandle>> matches;

  // Simple pattern matching with * wildcards
  auto matchPattern = [](const std::string &text, const std::string &pat) -> bool {
    size_t ti = 0, pi = 0;
    size_t starIdx = std::string::npos;
    size_t matchIdx = 0;

    while (ti < text.size()) {
      if (pi < pat.size() && (pat[pi] == '?' || pat[pi] == text[ti])) {
        ++pi;
        ++ti;
      } else if (pi < pat.size() && pat[pi] == '*') {
        starIdx = pi++;
        matchIdx = ti;
      } else if (starIdx != std::string::npos) {
        pi = starIdx + 1;
        ti = ++matchIdx;
      } else {
        return false;
      }
    }

    while (pi < pat.size() && pat[pi] == '*') {
      ++pi;
    }

    return pi == pat.size();
  };

  for (const auto &entry : signalRegistry.signals) {
    if (matchPattern(entry.first, pattern)) {
      matches.emplace_back(entry.first, entry.second.handle);
    }
  }

  return matches;
}
} // namespace

/// Look up a signal handle supporting hierarchical paths and wildcards.
extern "C" MooreSignalHandle __moore_signal_registry_lookup_hierarchical(
    const char *path) {
  if (!path || !*path)
    return MOORE_INVALID_SIGNAL_HANDLE;

  std::lock_guard<std::mutex> lock(signalRegistryMutex);
  return lookupSignalWithAlternatives(std::string(path));
}

/// Get a list of all registered signal paths.
/// Returns the count, and fills the buffer with paths (null-separated).
extern "C" uint64_t __moore_signal_registry_get_paths(char *buffer,
                                                       uint64_t bufferSize) {
  std::lock_guard<std::mutex> lock(signalRegistryMutex);

  uint64_t count = 0;
  uint64_t offset = 0;

  for (const auto &entry : signalRegistry.signals) {
    size_t pathLen = entry.first.size() + 1; // Include null terminator
    if (buffer && offset + pathLen <= bufferSize) {
      std::memcpy(buffer + offset, entry.first.c_str(), pathLen);
      offset += pathLen;
    }
    ++count;
  }

  return count;
}

/// Check if a signal is currently forced via DPI.
extern "C" int32_t __moore_signal_registry_is_forced(const char *path) {
  if (!path || !*path)
    return 0;

  std::lock_guard<std::mutex> lock(forcedSignalsMutex);
  auto it = forcedSignals.find(std::string(path));
  if (it != forcedSignals.end()) {
    return it->second.isForced ? 1 : 0;
  }
  return 0;
}

/// Get the forced value for a signal (if forced).
extern "C" int32_t __moore_signal_registry_get_forced_value(const char *path,
                                                             int64_t *value) {
  if (!path || !*path || !value)
    return 0;

  std::lock_guard<std::mutex> lock(forcedSignalsMutex);
  auto it = forcedSignals.find(std::string(path));
  if (it != forcedSignals.end() && it->second.isForced) {
    *value = it->second.forcedValue;
    return 1;
  }
  return 0;
}

/// Set a signal as forced with a specific value.
extern "C" int32_t __moore_signal_registry_set_forced(const char *path,
                                                       MooreSignalHandle handle,
                                                       int64_t value) {
  if (!path || !*path)
    return 0;

  std::lock_guard<std::mutex> lock(forcedSignalsMutex);
  forcedSignals[std::string(path)] = {handle, value, true};
  return 1;
}

/// Clear the forced state for a signal.
extern "C" int32_t __moore_signal_registry_clear_forced(const char *path) {
  if (!path || !*path)
    return 0;

  std::lock_guard<std::mutex> lock(forcedSignalsMutex);
  auto it = forcedSignals.find(std::string(path));
  if (it != forcedSignals.end()) {
    it->second.isForced = false;
    return 1;
  }
  return 0;
}

/// Clear all forced signals.
extern "C" void __moore_signal_registry_clear_all_forced(void) {
  std::lock_guard<std::mutex> lock(forcedSignalsMutex);
  forcedSignals.clear();
}

//===----------------------------------------------------------------------===//
// HDL Access Stubs
//===----------------------------------------------------------------------===//

namespace {
struct HDLValueEntry {
  uvm_hdl_data_t value = 0;
  bool forced = false;
};

std::unordered_map<std::string, HDLValueEntry> hdlValues;
std::mutex hdlMutex;

bool getPathKey(MooreString *path, std::string &key) {
  if (!path || !path->data || path->len <= 0)
    return false;
  key.assign(path->data, path->len);
  return !key.empty();
}

/// Helper to try reading from signal registry first, falling back to stub.
/// Uses hierarchical path lookup to find signals with various path formats.
bool tryRegistryRead(const std::string &key, uvm_hdl_data_t *value) {
  std::lock_guard<std::mutex> regLock(signalRegistryMutex);
  if (!signalRegistry.isConnected())
    return false;

  // First check if signal is forced - return forced value
  {
    std::lock_guard<std::mutex> forceLock(forcedSignalsMutex);
    auto forcedIt = forcedSignals.find(key);
    if (forcedIt != forcedSignals.end() && forcedIt->second.isForced) {
      *value = forcedIt->second.forcedValue;
      return true;
    }
  }

  // Try hierarchical lookup to find the signal
  MooreSignalHandle handle = lookupSignalWithAlternatives(key);
  if (handle == MOORE_INVALID_SIGNAL_HANDLE)
    return false;

  *value = signalRegistry.readCallback(handle, signalRegistry.userData);
  return true;
}

/// Helper to try writing to signal registry first, falling back to stub.
/// Uses hierarchical path lookup to find signals.
/// Respects force state - deposits are ignored if signal is forced.
bool tryRegistryWrite(const std::string &key, uvm_hdl_data_t value) {
  std::lock_guard<std::mutex> regLock(signalRegistryMutex);
  if (!signalRegistry.isConnected())
    return false;

  // Check if signal is forced - deposits are ignored for forced signals
  {
    std::lock_guard<std::mutex> forceLock(forcedSignalsMutex);
    auto forcedIt = forcedSignals.find(key);
    if (forcedIt != forcedSignals.end() && forcedIt->second.isForced) {
      // Signal is forced, deposit is ignored but return success
      return true;
    }
  }

  // Try hierarchical lookup to find the signal
  MooreSignalHandle handle = lookupSignalWithAlternatives(key);
  if (handle == MOORE_INVALID_SIGNAL_HANDLE)
    return false;

  signalRegistry.writeCallback(handle, value, signalRegistry.userData);
  return true;
}

/// Helper to try forcing via signal registry first, falling back to stub.
/// Uses hierarchical path lookup and updates force tracking.
bool tryRegistryForce(const std::string &key, uvm_hdl_data_t value) {
  std::lock_guard<std::mutex> regLock(signalRegistryMutex);
  if (!signalRegistry.isConnected() || !signalRegistry.forceCallback)
    return false;

  // Try hierarchical lookup to find the signal
  MooreSignalHandle handle = lookupSignalWithAlternatives(key);
  if (handle == MOORE_INVALID_SIGNAL_HANDLE)
    return false;

  // Track the forced state
  {
    std::lock_guard<std::mutex> forceLock(forcedSignalsMutex);
    forcedSignals[key] = {handle, value, true};
  }

  signalRegistry.forceCallback(handle, value, signalRegistry.userData);
  return true;
}

/// Helper to try releasing via signal registry first, falling back to stub.
/// Uses hierarchical path lookup and clears force tracking.
bool tryRegistryRelease(const std::string &key) {
  std::lock_guard<std::mutex> regLock(signalRegistryMutex);
  if (!signalRegistry.isConnected() || !signalRegistry.releaseCallback)
    return false;

  // Try hierarchical lookup to find the signal
  MooreSignalHandle handle = lookupSignalWithAlternatives(key);
  if (handle == MOORE_INVALID_SIGNAL_HANDLE)
    return false;

  // Clear the forced state
  {
    std::lock_guard<std::mutex> forceLock(forcedSignalsMutex);
    auto it = forcedSignals.find(key);
    if (it != forcedSignals.end()) {
      it->second.isForced = false;
    }
  }

  signalRegistry.releaseCallback(handle, signalRegistry.userData);
  return true;
}
} // namespace

extern "C" int32_t uvm_hdl_check_path(MooreString *path) {
  std::string key;
  if (!getPathKey(path, key))
    return 0;

  // Check signal registry first using hierarchical lookup
  {
    std::lock_guard<std::mutex> regLock(signalRegistryMutex);
    MooreSignalHandle handle = lookupSignalWithAlternatives(key);
    if (handle != MOORE_INVALID_SIGNAL_HANDLE)
      return 1;
  }

  // Fall back to stub map - create entry if not exists
  std::lock_guard<std::mutex> lock(hdlMutex);
  (void)hdlValues[key];
  return 1;
}

extern "C" int32_t uvm_hdl_deposit(MooreString *path, uvm_hdl_data_t value) {
  std::string key;
  if (!getPathKey(path, key))
    return 0;

  // Try signal registry first (actual simulation signals)
  // Note: Check hdlValues first for forced status
  {
    std::lock_guard<std::mutex> lock(hdlMutex);
    auto it = hdlValues.find(key);
    if (it != hdlValues.end() && it->second.forced) {
      // Signal is forced, deposit is ignored
      return 1;
    }
  }

  // Try to deposit via registry
  if (tryRegistryWrite(key, value)) {
    // Also update stub map to keep them in sync for VPI access
    std::lock_guard<std::mutex> lock(hdlMutex);
    hdlValues[key].value = value;
    return 1;
  }

  // Fall back to stub map
  std::lock_guard<std::mutex> lock(hdlMutex);
  auto &entry = hdlValues[key];
  if (!entry.forced)
    entry.value = value;
  return 1;
}

extern "C" int32_t uvm_hdl_force(MooreString *path, uvm_hdl_data_t value) {
  std::string key;
  if (!getPathKey(path, key))
    return 0;

  // Try signal registry first (actual simulation signals)
  if (tryRegistryForce(key, value)) {
    // Also update stub map to track forced status
    std::lock_guard<std::mutex> lock(hdlMutex);
    auto &entry = hdlValues[key];
    entry.value = value;
    entry.forced = true;
    return 1;
  }

  // Fall back to stub map
  std::lock_guard<std::mutex> lock(hdlMutex);
  auto &entry = hdlValues[key];
  entry.value = value;
  entry.forced = true;
  return 1;
}

extern "C" int32_t uvm_hdl_release_and_read(MooreString *path,
                                             uvm_hdl_data_t *value) {
  std::string key;
  if (!value || !getPathKey(path, key))
    return 0;

  // Try signal registry first (actual simulation signals)
  if (tryRegistryRelease(key)) {
    // Clear forced status in stub map
    {
      std::lock_guard<std::mutex> lock(hdlMutex);
      auto it = hdlValues.find(key);
      if (it != hdlValues.end())
        it->second.forced = false;
    }
    // Read the released value
    if (tryRegistryRead(key, value))
      return 1;
  }

  // Fall back to stub map
  std::lock_guard<std::mutex> lock(hdlMutex);
  auto it = hdlValues.find(key);
  if (it != hdlValues.end()) {
    it->second.forced = false;
    *value = it->second.value;
  } else {
    *value = 0;
  }
  return 1;
}

extern "C" int32_t uvm_hdl_release(MooreString *path) {
  std::string key;
  if (!getPathKey(path, key))
    return 0;

  // Try signal registry first (actual simulation signals)
  if (tryRegistryRelease(key)) {
    // Also clear forced status in stub map
    std::lock_guard<std::mutex> lock(hdlMutex);
    auto it = hdlValues.find(key);
    if (it != hdlValues.end())
      it->second.forced = false;
    return 1;
  }

  // Fall back to stub map
  std::lock_guard<std::mutex> lock(hdlMutex);
  auto it = hdlValues.find(key);
  if (it != hdlValues.end())
    it->second.forced = false;
  return 1;
}

extern "C" int32_t uvm_hdl_read(MooreString *path, uvm_hdl_data_t *value) {
  std::string key;
  if (!value || !getPathKey(path, key))
    return 0;

  // Try signal registry first (actual simulation signals)
  if (tryRegistryRead(key, value))
    return 1;

  // Fall back to stub map
  std::lock_guard<std::mutex> lock(hdlMutex);
  auto it = hdlValues.find(key);
  if (it != hdlValues.end())
    *value = it->second.value;
  else
    *value = 0;
  return 1;
}

//===----------------------------------------------------------------------===//
// Regular Expression Stubs
//===----------------------------------------------------------------------===//

namespace {
// Simple regex stub: stores the pattern string for minimal matching.
struct UVMRegexStub {
  std::string pattern;
  bool deglob;
  bool valid = false;
};

// Buffer for last match result (used by uvm_re_buffer)
std::string lastMatchBuffer;
} // namespace

extern "C" void *uvm_re_comp(MooreString *pattern, int32_t deglob) {
  if (!pattern || !pattern->data) {
    return nullptr;
  }

  std::string regexPattern(pattern->data, pattern->len);
  const bool useGlob = (deglob != 0);
  if (useGlob) {
    regexPattern = convertGlobToRegex(regexPattern, true);
  }

  auto isSupportedPattern = [](const std::string &pat) -> bool {
    for (size_t i = 0; i < pat.size(); ++i) {
      char c = pat[i];
      if (c == '\\') {
        if (i + 1 < pat.size())
          ++i;
        continue;
      }
      if (c == '[' || c == ']')
        return false;
    }
    return true;
  };

  if (!isSupportedPattern(regexPattern))
    return nullptr;

  auto *stub = new UVMRegexStub();
  stub->pattern = regexPattern;
  stub->deglob = useGlob;
  stub->valid = true;

  // Optional: Print debug info
  // std::printf("[DPI] uvm_re_comp: %s (deglob=%d)\n",
  //             stub->pattern.c_str(), deglob);

  return static_cast<void *>(stub);
}

extern "C" int32_t uvm_re_exec(void *rexp, MooreString *str) {
  if (!rexp || !str || !str->data) {
    return -1; // No match
  }

  auto *stub = static_cast<UVMRegexStub *>(rexp);
  if (!stub->valid)
    return -1;
  std::string target(str->data, str->len);

  auto readToken = [](const std::string &pat, size_t idx, char &tok, bool &any,
                      size_t &nextIdx) -> bool {
    if (idx >= pat.size())
      return false;
    if (pat[idx] == '\\' && idx + 1 < pat.size()) {
      tok = pat[idx + 1];
      any = false;
      nextIdx = idx + 2;
      return true;
    }
    if (pat[idx] == '.') {
      tok = 0;
      any = true;
      nextIdx = idx + 1;
      return true;
    }
    tok = pat[idx];
    any = false;
    nextIdx = idx + 1;
    return true;
  };

  std::function<bool(size_t, size_t, size_t &)> matchFrom =
      [&](size_t patIdx, size_t strIdx, size_t &endIdx) -> bool {
    if (patIdx >= stub->pattern.size()) {
      endIdx = strIdx;
      return true;
    }
    char tok = 0;
    bool any = false;
    size_t nextIdx = 0;
    if (!readToken(stub->pattern, patIdx, tok, any, nextIdx))
      return false;
    bool hasStar =
        (nextIdx < stub->pattern.size() && stub->pattern[nextIdx] == '*');
    if (hasStar) {
      size_t nextPat = nextIdx + 1;
      if (matchFrom(nextPat, strIdx, endIdx))
        return true;
      size_t i = strIdx;
      while (i < target.size() && (any || target[i] == tok)) {
        ++i;
        if (matchFrom(nextPat, i, endIdx))
          return true;
      }
      return false;
    }
    if (strIdx >= target.size())
      return false;
    if (any || target[strIdx] == tok)
      return matchFrom(nextIdx, strIdx + 1, endIdx);
    return false;
  };

  for (size_t start = 0; start <= target.size(); ++start) {
    size_t endIdx = 0;
    if (matchFrom(0, start, endIdx)) {
      lastMatchBuffer = target.substr(start, endIdx - start);
      return static_cast<int32_t>(start);
    }
  }

  lastMatchBuffer.clear();
  return -1; // No match
}

extern "C" void uvm_re_free(void *rexp) {
  if (rexp) {
    auto *stub = static_cast<UVMRegexStub *>(rexp);
    delete stub;
  }
}

extern "C" MooreString uvm_re_buffer(void) {
  // Return the last matched substring
  MooreString result;
  if (lastMatchBuffer.empty()) {
    result.data = nullptr;
    result.len = 0;
  } else {
    result.len = static_cast<int64_t>(lastMatchBuffer.size());
    result.data = static_cast<char *>(std::malloc(result.len));
    if (result.data) {
      std::memcpy(result.data, lastMatchBuffer.data(), result.len);
    } else {
      result.len = 0;
    }
  }
  return result;
}

extern "C" int32_t uvm_re_compexecfree(MooreString *pattern, MooreString *str,
                                        int32_t deglob, int32_t *exec_ret) {
  if (!pattern || !str || !exec_ret) {
    if (exec_ret)
      *exec_ret = -1;
    return 0; // Invalid regex
  }

  // Compile, execute, and free in one go
  void *rexp = uvm_re_comp(pattern, deglob);
  if (!rexp) {
    *exec_ret = -1;
    return 0; // Invalid regex
  }

  *exec_ret = uvm_re_exec(rexp, str);
  uvm_re_free(rexp);

  return 1; // Valid regex
}

extern "C" MooreString uvm_re_deglobbed(MooreString *glob,
                                         int32_t with_brackets) {
  if (!glob || !glob->data) {
    MooreString result = {nullptr, 0};
    return result;
  }

  std::string pattern(glob->data, glob->len);
  std::string regex =
      convertGlobToRegex(pattern, with_brackets != 0);

  // Allocate and return result
  MooreString result;
  result.len = static_cast<int64_t>(regex.size());
  result.data = static_cast<char *>(std::malloc(result.len));
  if (result.data) {
    std::memcpy(result.data, regex.data(), result.len);
  } else {
    result.len = 0;
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Command Line / Tool Info Stubs
//===----------------------------------------------------------------------===//

namespace {
// Simulated command line arguments (empty for stub)
std::vector<std::string> cmdLineArgs;
std::string cmdLineArgsEnv;

void parseCommandLineArgs(const std::string &args) {
  cmdLineArgs.clear();

  std::string current;
  bool inQuotes = false;
  char quoteChar = '\0';
  for (size_t i = 0; i < args.size(); ++i) {
    char c = args[i];
    if (c == '\\' && i + 1 < args.size()) {
      char next = args[i + 1];
      if (next == '"' || next == '\'' || next == '\\') {
        current.push_back(next);
        ++i;
        continue;
      }
    }
    if ((c == '"' || c == '\'') && (!inQuotes || c == quoteChar)) {
      if (inQuotes) {
        inQuotes = false;
        quoteChar = '\0';
      } else {
        inQuotes = true;
        quoteChar = c;
      }
      continue;
    }
    if (!inQuotes && std::isspace(static_cast<unsigned char>(c))) {
      if (!current.empty()) {
        cmdLineArgs.push_back(current);
        current.clear();
      }
      continue;
    }
    current.push_back(c);
  }
  if (!current.empty())
    cmdLineArgs.push_back(current);
}

void initCommandLineArgs() {
  const char *env = std::getenv("CIRCT_UVM_ARGS");
  if (!env)
    env = std::getenv("UVM_ARGS");
  std::string nextEnv = env ? std::string(env) : std::string();

  if (nextEnv == cmdLineArgsEnv)
    return;

  cmdLineArgsEnv = nextEnv;
  if (cmdLineArgsEnv.empty())
    cmdLineArgs.clear();
  else
    parseCommandLineArgs(cmdLineArgsEnv);
}
} // namespace

// Static index for iterating command line arguments
static int32_t uvmCmdLineIdx = 0;

extern "C" MooreString uvm_dpi_get_next_arg_c(int32_t init) {
  initCommandLineArgs();

  // UVM DPI spec: init=1 resets to start, init=0 gets next arg
  if (init) {
    uvmCmdLineIdx = 0;
  }

  // Return empty string when no more arguments
  if (uvmCmdLineIdx >= static_cast<int32_t>(cmdLineArgs.size())) {
    MooreString result = {nullptr, 0};
    return result;
  }

  const std::string &arg = cmdLineArgs[uvmCmdLineIdx];
  uvmCmdLineIdx++;

  MooreString result;
  result.len = static_cast<int64_t>(arg.size());
  result.data = static_cast<char *>(std::malloc(result.len));
  if (result.data) {
    std::memcpy(result.data, arg.data(), result.len);
  } else {
    result.len = 0;
  }

  return result;
}

extern "C" MooreString uvm_dpi_get_tool_name_c(void) {
  // Return "CIRCT" as the tool name
  const char *toolName = "CIRCT";
  size_t len = std::strlen(toolName);

  MooreString result;
  result.len = static_cast<int64_t>(len);
  result.data = static_cast<char *>(std::malloc(len));
  if (result.data) {
    std::memcpy(result.data, toolName, len);
  } else {
    result.len = 0;
  }

  return result;
}

extern "C" MooreString uvm_dpi_get_tool_version_c(void) {
  // Return "1.0" as the tool version
  const char *toolVersion = "1.0";
  size_t len = std::strlen(toolVersion);

  MooreString result;
  result.len = static_cast<int64_t>(len);
  result.data = static_cast<char *>(std::malloc(len));
  if (result.data) {
    std::memcpy(result.data, toolVersion, len);
  } else {
    result.len = 0;
  }

  return result;
}

/// Parse +UVM_TESTNAME from command-line arguments.
/// This function searches through all command-line arguments for a
/// +UVM_TESTNAME=<name> argument. The argument format follows the standard
/// UVM command-line processing format.
///
/// @return A MooreString containing the test name if found, or an empty
///         MooreString if not found.
extern "C" MooreString __moore_uvm_get_testname_from_cmdline(void) {
  // Initialize command line args from environment
  initCommandLineArgs();

  MooreString result = {nullptr, 0};

  // The prefix we're looking for
  const char *prefix = "+UVM_TESTNAME=";
  size_t prefixLen = std::strlen(prefix);

  // Search through all command line arguments
  for (const std::string &arg : cmdLineArgs) {
    // Check if the argument starts with +UVM_TESTNAME=
    if (arg.size() > prefixLen &&
        arg.compare(0, prefixLen, prefix) == 0) {
      // Extract the test name (everything after the '=')
      std::string testName = arg.substr(prefixLen);

      // Return the test name
      result.len = static_cast<int64_t>(testName.size());
      if (result.len > 0) {
        result.data = static_cast<char *>(std::malloc(result.len));
        if (result.data) {
          std::memcpy(result.data, testName.data(), result.len);
        } else {
          result.len = 0;
        }
      }
      return result;
    }
  }

  // Not found - return empty string
  return result;
}

/// Check if +UVM_TESTNAME was specified on the command line.
/// @return 1 if found, 0 otherwise
extern "C" int32_t __moore_uvm_has_cmdline_testname(void) {
  // Initialize command line args from environment
  initCommandLineArgs();

  // The prefix we're looking for
  const char *prefix = "+UVM_TESTNAME=";
  size_t prefixLen = std::strlen(prefix);

  // Search through all command line arguments
  for (const std::string &arg : cmdLineArgs) {
    if (arg.size() > prefixLen &&
        arg.compare(0, prefixLen, prefix) == 0) {
      return 1;
    }
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// UVM Coverage Model API
//===----------------------------------------------------------------------===//
//
// Implementation of UVM-compatible coverage API for register and field coverage.
// This integrates with the existing covergroup infrastructure to provide
// UVM-style coverage collection tied to register and field access patterns.
//

namespace {

/// State for UVM coverage tracking.
struct UvmCoverageState {
  /// Current coverage model bitmask.
  int32_t coverageModel = UVM_NO_COVERAGE;

  /// Map from register names to their covergroup handles.
  std::unordered_map<std::string, void *> regCovergroups;

  /// Map from field names to their covergroup handles.
  std::unordered_map<std::string, void *> fieldCovergroups;

  /// Map from address map names to their covergroup handles.
  std::unordered_map<std::string, void *> addrMapCovergroups;

  /// Map from register names to their bit widths (default: 64).
  std::unordered_map<std::string, int32_t> regBitWidths;

  /// Map from field names to their value ranges.
  std::unordered_map<std::string, std::pair<int64_t, int64_t>> fieldRanges;

  /// Register coverage callback.
  MooreUvmRegCoverageCallback regCallback = nullptr;
  void *regCallbackUserData = nullptr;

  /// Field coverage callback.
  MooreUvmFieldCoverageCallback fieldCallback = nullptr;
  void *fieldCallbackUserData = nullptr;

  /// Mutex for thread safety.
  std::mutex mutex;
};

/// Global UVM coverage state.
UvmCoverageState uvmCoverageState;

/// Get or create a covergroup for a register.
void *getOrCreateRegCovergroup(const std::string &regName) {
  auto it = uvmCoverageState.regCovergroups.find(regName);
  if (it != uvmCoverageState.regCovergroups.end()) {
    return it->second;
  }

  // Create a covergroup with a single coverpoint for the register.
  std::string cgName = "uvm_reg_" + regName;
  void *cg = __moore_covergroup_create(cgName.c_str(), 1);
  if (!cg)
    return nullptr;

  // Get bit width for this register (default: 8 for manageable auto bins).
  int32_t bitWidth = 8;
  auto bitWidthIt = uvmCoverageState.regBitWidths.find(regName);
  if (bitWidthIt != uvmCoverageState.regBitWidths.end()) {
    bitWidth = bitWidthIt->second;
  }

  // Initialize the coverpoint with auto bins.
  // Auto bins track which values have been seen within the observed range.
  __moore_coverpoint_init(cg, 0, regName.c_str());

  // Set auto_bin_max based on bit width to limit bin count.
  int64_t autoBinMax = std::min(static_cast<int64_t>(64),
                                static_cast<int64_t>(1) << bitWidth);
  __moore_coverpoint_set_auto_bin_max(cg, 0, autoBinMax);

  uvmCoverageState.regCovergroups[regName] = cg;
  return cg;
}

/// Get or create a covergroup for a field.
void *getOrCreateFieldCovergroup(const std::string &fieldName) {
  auto it = uvmCoverageState.fieldCovergroups.find(fieldName);
  if (it != uvmCoverageState.fieldCovergroups.end()) {
    return it->second;
  }

  // Create a covergroup with a single coverpoint for the field.
  std::string cgName = "uvm_field_" + fieldName;
  void *cg = __moore_covergroup_create(cgName.c_str(), 1);
  if (!cg)
    return nullptr;

  // Get field range (default: 0-255 for 8-bit field).
  int64_t minVal = 0;
  int64_t maxVal = 255;
  auto rangeIt = uvmCoverageState.fieldRanges.find(fieldName);
  if (rangeIt != uvmCoverageState.fieldRanges.end()) {
    minVal = rangeIt->second.first;
    maxVal = rangeIt->second.second;
  }

  // Initialize the coverpoint with auto bins.
  __moore_coverpoint_init(cg, 0, fieldName.c_str());

  // Set auto_bin_max based on range size.
  int64_t rangeSize = maxVal - minVal + 1;
  int64_t autoBinMax = std::min(static_cast<int64_t>(64), rangeSize);
  __moore_coverpoint_set_auto_bin_max(cg, 0, autoBinMax);

  uvmCoverageState.fieldCovergroups[fieldName] = cg;
  return cg;
}

/// Get or create a covergroup for an address map.
void *getOrCreateAddrMapCovergroup(const std::string &mapName) {
  auto it = uvmCoverageState.addrMapCovergroups.find(mapName);
  if (it != uvmCoverageState.addrMapCovergroups.end()) {
    return it->second;
  }

  // Create a covergroup with two coverpoints: address and access type.
  std::string cgName = "uvm_addr_map_" + mapName;
  void *cg = __moore_covergroup_create(cgName.c_str(), 2);
  if (!cg)
    return nullptr;

  // Initialize address coverpoint (auto bins for address range).
  std::string addrCpName = mapName + "_addr";
  __moore_coverpoint_init(cg, 0, addrCpName.c_str());
  __moore_coverpoint_set_auto_bin_max(cg, 0, 64);

  // Initialize access type coverpoint (read=1, write=0).
  std::string accessCpName = mapName + "_access";
  __moore_coverpoint_init(cg, 1, accessCpName.c_str());
  __moore_coverpoint_set_auto_bin_max(cg, 1, 2);  // Only 2 values: read/write

  uvmCoverageState.addrMapCovergroups[mapName] = cg;
  return cg;
}

} // anonymous namespace

extern "C" void __moore_uvm_set_coverage_model(int32_t model) {
  std::lock_guard<std::mutex> lock(uvmCoverageState.mutex);
  uvmCoverageState.coverageModel = model;
}

extern "C" int32_t __moore_uvm_get_coverage_model(void) {
  std::lock_guard<std::mutex> lock(uvmCoverageState.mutex);
  return uvmCoverageState.coverageModel;
}

extern "C" bool __moore_uvm_has_coverage(int32_t model) {
  std::lock_guard<std::mutex> lock(uvmCoverageState.mutex);
  return (uvmCoverageState.coverageModel & model) != 0;
}

extern "C" void __moore_uvm_coverage_sample_reg(const char *reg_name,
                                                 int64_t value) {
  if (!reg_name)
    return;

  std::lock_guard<std::mutex> lock(uvmCoverageState.mutex);

  // Only sample if UVM_CVR_REG_BITS is enabled.
  if ((uvmCoverageState.coverageModel & UVM_CVR_REG_BITS) == 0)
    return;

  // Invoke callback if registered.
  if (uvmCoverageState.regCallback) {
    uvmCoverageState.regCallback(reg_name, value,
                                 uvmCoverageState.regCallbackUserData);
  }

  // Get or create the covergroup for this register.
  void *cg = getOrCreateRegCovergroup(reg_name);
  if (!cg)
    return;

  // Sample the value into the coverpoint.
  __moore_coverpoint_sample(cg, 0, value);
}

extern "C" void __moore_uvm_coverage_sample_field(const char *field_name,
                                                   int64_t value) {
  if (!field_name)
    return;

  std::lock_guard<std::mutex> lock(uvmCoverageState.mutex);

  // Only sample if UVM_CVR_FIELD_VALS is enabled.
  if ((uvmCoverageState.coverageModel & UVM_CVR_FIELD_VALS) == 0)
    return;

  // Invoke callback if registered.
  if (uvmCoverageState.fieldCallback) {
    uvmCoverageState.fieldCallback(field_name, value,
                                   uvmCoverageState.fieldCallbackUserData);
  }

  // Get or create the covergroup for this field.
  void *cg = getOrCreateFieldCovergroup(field_name);
  if (!cg)
    return;

  // Sample the value into the coverpoint.
  __moore_coverpoint_sample(cg, 0, value);
}

extern "C" void __moore_uvm_coverage_sample_addr_map(const char *map_name,
                                                      int64_t address,
                                                      bool is_read) {
  if (!map_name)
    return;

  std::lock_guard<std::mutex> lock(uvmCoverageState.mutex);

  // Only sample if UVM_CVR_ADDR_MAP is enabled.
  if ((uvmCoverageState.coverageModel & UVM_CVR_ADDR_MAP) == 0)
    return;

  // Get or create the covergroup for this address map.
  void *cg = getOrCreateAddrMapCovergroup(map_name);
  if (!cg)
    return;

  // Sample the address and access type.
  __moore_coverpoint_sample(cg, 0, address);
  __moore_coverpoint_sample(cg, 1, is_read ? 1 : 0);
}

extern "C" double __moore_uvm_get_reg_coverage(const char *reg_name) {
  if (!reg_name)
    return 0.0;

  std::lock_guard<std::mutex> lock(uvmCoverageState.mutex);

  auto it = uvmCoverageState.regCovergroups.find(reg_name);
  if (it == uvmCoverageState.regCovergroups.end()) {
    return 0.0; // No coverage data for this register.
  }

  return __moore_covergroup_get_coverage(it->second);
}

extern "C" double __moore_uvm_get_field_coverage(const char *field_name) {
  if (!field_name)
    return 0.0;

  std::lock_guard<std::mutex> lock(uvmCoverageState.mutex);

  auto it = uvmCoverageState.fieldCovergroups.find(field_name);
  if (it == uvmCoverageState.fieldCovergroups.end()) {
    return 0.0; // No coverage data for this field.
  }

  return __moore_covergroup_get_coverage(it->second);
}

extern "C" double __moore_uvm_get_coverage(void) {
  std::lock_guard<std::mutex> lock(uvmCoverageState.mutex);

  double totalCoverage = 0.0;
  int count = 0;

  // Aggregate coverage from all register covergroups.
  for (const auto &pair : uvmCoverageState.regCovergroups) {
    totalCoverage += __moore_covergroup_get_coverage(pair.second);
    count++;
  }

  // Aggregate coverage from all field covergroups.
  for (const auto &pair : uvmCoverageState.fieldCovergroups) {
    totalCoverage += __moore_covergroup_get_coverage(pair.second);
    count++;
  }

  // Aggregate coverage from all address map covergroups.
  for (const auto &pair : uvmCoverageState.addrMapCovergroups) {
    totalCoverage += __moore_covergroup_get_coverage(pair.second);
    count++;
  }

  if (count == 0)
    return 0.0;

  return totalCoverage / count;
}

extern "C" void __moore_uvm_reset_coverage(void) {
  std::lock_guard<std::mutex> lock(uvmCoverageState.mutex);

  // Reset all register covergroups.
  for (const auto &pair : uvmCoverageState.regCovergroups) {
    __moore_covergroup_reset(pair.second);
  }

  // Reset all field covergroups.
  for (const auto &pair : uvmCoverageState.fieldCovergroups) {
    __moore_covergroup_reset(pair.second);
  }

  // Reset all address map covergroups.
  for (const auto &pair : uvmCoverageState.addrMapCovergroups) {
    __moore_covergroup_reset(pair.second);
  }
}

extern "C" void __moore_uvm_set_reg_coverage_callback(
    MooreUvmRegCoverageCallback callback, void *userData) {
  std::lock_guard<std::mutex> lock(uvmCoverageState.mutex);
  uvmCoverageState.regCallback = callback;
  uvmCoverageState.regCallbackUserData = userData;
}

extern "C" void __moore_uvm_set_field_coverage_callback(
    MooreUvmFieldCoverageCallback callback, void *userData) {
  std::lock_guard<std::mutex> lock(uvmCoverageState.mutex);
  uvmCoverageState.fieldCallback = callback;
  uvmCoverageState.fieldCallbackUserData = userData;
}

extern "C" void __moore_uvm_set_reg_bit_width(const char *reg_name,
                                               int32_t bit_width) {
  if (!reg_name || bit_width < 1 || bit_width > 64)
    return;

  std::lock_guard<std::mutex> lock(uvmCoverageState.mutex);
  uvmCoverageState.regBitWidths[reg_name] = bit_width;

  // If the covergroup already exists, update its auto_bin_max.
  auto it = uvmCoverageState.regCovergroups.find(reg_name);
  if (it != uvmCoverageState.regCovergroups.end()) {
    int64_t autoBinMax = std::min(static_cast<int64_t>(64),
                                  static_cast<int64_t>(1) << bit_width);
    __moore_coverpoint_set_auto_bin_max(it->second, 0, autoBinMax);
  }
}

extern "C" void __moore_uvm_set_field_range(const char *field_name,
                                             int64_t min_val, int64_t max_val) {
  if (!field_name || min_val > max_val)
    return;

  std::lock_guard<std::mutex> lock(uvmCoverageState.mutex);
  uvmCoverageState.fieldRanges[field_name] = {min_val, max_val};

  // Note: If the covergroup already exists, the range won't be updated.
  // Users should set the range before sampling.
}

//===----------------------------------------------------------------------===//
// VPI Stub Support
//===----------------------------------------------------------------------===//

namespace {
struct VpiHandleImpl {
  std::string name;
};

thread_local std::string vpiStringResult;

void releaseVpiHandle(vpiHandle handle) {
  if (!handle)
    return;
  auto *impl = static_cast<VpiHandleImpl *>(handle);
  delete impl;
}
} // namespace

extern "C" vpiHandle vpi_handle_by_name(const char *name, vpiHandle scope) {
  (void)scope;
  if (!name || !*name)
    return nullptr;

  // Initialize in stub map (for compatibility)
  {
    std::lock_guard<std::mutex> lock(hdlMutex);
    (void)hdlValues[std::string(name)];
  }

  auto *handle = new VpiHandleImpl();
  handle->name = name;
  return static_cast<vpiHandle>(handle);
}

extern "C" int32_t vpi_get(int32_t property, vpiHandle obj) {
  (void)property;
  return obj ? 1 : 0;
}

extern "C" char *vpi_get_str(int32_t property, vpiHandle obj) {
  (void)property;
  if (!obj)
    return nullptr;
  auto *handle = static_cast<VpiHandleImpl *>(obj);
  vpiStringResult = handle->name;
  return const_cast<char *>(vpiStringResult.c_str());
}

extern "C" void vpi_release_handle(vpiHandle obj) {
  releaseVpiHandle(obj);
}

extern "C" int32_t vpi_get_value(vpiHandle obj, vpi_value *value) {
  if (!obj || !value || !value->value)
    return 0;
  auto *handle = static_cast<VpiHandleImpl *>(obj);

  // Try signal registry first (actual simulation signals)
  uvm_hdl_data_t readValue = 0;
  if (tryRegistryRead(handle->name, &readValue)) {
    *static_cast<uvm_hdl_data_t *>(value->value) = readValue;
    return 1;
  }

  // Fall back to stub map
  std::lock_guard<std::mutex> lock(hdlMutex);
  auto it = hdlValues.find(handle->name);
  if (it == hdlValues.end()) {
    *static_cast<uvm_hdl_data_t *>(value->value) = 0;
  } else {
    *static_cast<uvm_hdl_data_t *>(value->value) = it->second.value;
  }
  return 1;
}

extern "C" int32_t vpi_put_value(vpiHandle obj, vpi_value *value, void *time,
                                 int32_t flags) {
  (void)time;
  if (!obj || !value || !value->value)
    return 0;
  auto *handle = static_cast<VpiHandleImpl *>(obj);
  uvm_hdl_data_t writeValue = *static_cast<uvm_hdl_data_t *>(value->value);
  bool isForce = (flags != 0);

  // Try signal registry first (actual simulation signals)
  if (isForce) {
    if (tryRegistryForce(handle->name, writeValue)) {
      // Also update stub map to track forced status
      std::lock_guard<std::mutex> lock(hdlMutex);
      auto &entry = hdlValues[handle->name];
      entry.value = writeValue;
      entry.forced = true;
      return 1;
    }
  } else {
    // Check if forced
    {
      std::lock_guard<std::mutex> lock(hdlMutex);
      auto it = hdlValues.find(handle->name);
      if (it != hdlValues.end() && it->second.forced) {
        // Signal is forced, normal write is ignored
        return 1;
      }
    }
    if (tryRegistryWrite(handle->name, writeValue)) {
      // Also update stub map
      std::lock_guard<std::mutex> lock(hdlMutex);
      hdlValues[handle->name].value = writeValue;
      return 1;
    }
  }

  // Fall back to stub map
  std::lock_guard<std::mutex> lock(hdlMutex);
  auto &entry = hdlValues[handle->name];
  entry.value = writeValue;
  entry.forced = isForce;
  return 1;
}

//===----------------------------------------------------------------------===//
// UVM Configuration Database
//===----------------------------------------------------------------------===//

namespace {

/// Entry structure for config_db storage.
/// Stores a copy of the value and its type ID for type checking.
struct ConfigDbEntry {
  std::string instName;        // Original instance name (may contain wildcards)
  std::string fieldName;       // Field name
  std::vector<uint8_t> value;  // Deep copy of the stored value
  int32_t typeId;
  uint64_t setTime;            // Monotonic counter for last-set-wins ordering
  bool hasWildcard;            // True if instName contains wildcard characters
};

/// Global config_db storage, keyed by "{inst_name}.{field_name}".
/// Thread-safe via mutex protection.
std::unordered_map<std::string, ConfigDbEntry> __moore_config_db_storage;
std::mutex __moore_config_db_mutex;
std::atomic<uint64_t> __moore_config_db_set_counter{0};

/// Check if a string contains wildcard characters (* or ?).
bool containsWildcard(const std::string &str) {
  return str.find('*') != std::string::npos ||
         str.find('?') != std::string::npos;
}

/// Check if a glob pattern matches a path.
/// Supports * (match any characters) and ? (match single character).
bool matchesGlobPattern(const std::string &pattern, const std::string &path) {
  // Empty pattern matches empty path
  if (pattern.empty())
    return path.empty();

  // Pattern "*" matches everything
  if (pattern == "*")
    return true;

  // Convert glob to regex and match
  std::string regexPattern = "^" + convertGlobToRegex(pattern, false) + "$";
  std::regex re(regexPattern, std::regex::nosubs | std::regex::optimize);
  return std::regex_match(path, re);
}

/// Check if setPath (from config_db_set) can provide a value for lookupPath (from get).
/// This implements hierarchical matching where parent paths match child paths.
/// For example: set("top.*", ...) should match get("top.env.agent", ...).
bool pathMatches(const std::string &setPath, const std::string &lookupPath) {
  // Exact match
  if (setPath == lookupPath)
    return true;

  // If setPath contains wildcards, use glob matching
  if (containsWildcard(setPath)) {
    return matchesGlobPattern(setPath, lookupPath);
  }

  // Hierarchical matching: setPath is a prefix of lookupPath
  // e.g., setPath="top.env" should match lookupPath="top.env.agent"
  if (!setPath.empty() && lookupPath.size() > setPath.size()) {
    if (lookupPath.compare(0, setPath.size(), setPath) == 0 &&
        lookupPath[setPath.size()] == '.') {
      return true;
    }
  }

  return false;
}

/// Build the config_db key from instance name and field name.
std::string buildConfigDbKey(const char *instName, int64_t instLen,
                             const char *fieldName, int64_t fieldLen) {
  std::string key;
  if (instName && instLen > 0) {
    key.append(instName, static_cast<size_t>(instLen));
  }
  key.push_back('.');
  if (fieldName && fieldLen > 0) {
    key.append(fieldName, static_cast<size_t>(fieldLen));
  }
  return key;
}

/// Convert instance name and field name to strings.
std::pair<std::string, std::string> extractNames(const char *instName,
                                                  int64_t instLen,
                                                  const char *fieldName,
                                                  int64_t fieldLen) {
  std::string inst = (instName && instLen > 0)
                         ? std::string(instName, static_cast<size_t>(instLen))
                         : "";
  std::string field = (fieldName && fieldLen > 0)
                          ? std::string(fieldName, static_cast<size_t>(fieldLen))
                          : "";
  return {inst, field};
}

} // anonymous namespace

/// Set a value in the UVM configuration database.
/// @param context Pointer to the context (currently unused, for future hierarchy support)
/// @param instName Pointer to the instance name string data
/// @param instLen Length of the instance name string
/// @param fieldName Pointer to the field name string data
/// @param fieldLen Length of the field name string
/// @param value Pointer to the value to store
/// @param valueSize Size of the value in bytes
/// @param typeId Type identifier for type checking on retrieval
extern "C" void __moore_config_db_set(void *context, const char *instName,
                                      int64_t instLen, const char *fieldName,
                                      int64_t fieldLen, void *value,
                                      int64_t valueSize, int32_t typeId) {
  (void)context;  // Reserved for future use (UVM hierarchy context)

  auto [inst, field] = extractNames(instName, instLen, fieldName, fieldLen);
  std::string key = buildConfigDbKey(instName, instLen, fieldName, fieldLen);

  ConfigDbEntry entry;
  entry.instName = inst;
  entry.fieldName = field;
  entry.typeId = typeId;
  entry.setTime = __moore_config_db_set_counter.fetch_add(1);
  entry.hasWildcard = containsWildcard(inst);
  if (value && valueSize > 0) {
    entry.value.resize(static_cast<size_t>(valueSize));
    std::memcpy(entry.value.data(), value, static_cast<size_t>(valueSize));
  }

  std::lock_guard<std::mutex> lock(__moore_config_db_mutex);
  __moore_config_db_storage[key] = std::move(entry);
}

/// Get a value from the UVM configuration database.
/// @param context Pointer to the context (currently unused, for future hierarchy support)
/// @param instName Pointer to the instance name string data
/// @param instLen Length of the instance name string
/// @param fieldName Pointer to the field name string data
/// @param fieldLen Length of the field name string
/// @param typeId Expected type identifier for type checking
/// @param outValue Pointer to the output buffer for the value
/// @param valueSize Size of the output buffer in bytes
/// @return 1 if the value was found and types match, 0 otherwise
extern "C" int32_t __moore_config_db_get(void *context, const char *instName,
                                         int64_t instLen, const char *fieldName,
                                         int64_t fieldLen, int32_t typeId,
                                         void *outValue, int64_t valueSize) {
  (void)context;  // Reserved for future use (UVM hierarchy context)

  auto [lookupInst, lookupField] = extractNames(instName, instLen, fieldName, fieldLen);
  std::string exactKey = buildConfigDbKey(instName, instLen, fieldName, fieldLen);

  std::lock_guard<std::mutex> lock(__moore_config_db_mutex);

  // 1. Try exact match first (most common case)
  auto exactIt = __moore_config_db_storage.find(exactKey);
  if (exactIt != __moore_config_db_storage.end()) {
    const ConfigDbEntry &entry = exactIt->second;
    // Copy the value to the output buffer
    if (outValue && valueSize > 0 && !entry.value.empty()) {
      size_t copySize = std::min(static_cast<size_t>(valueSize), entry.value.size());
      std::memcpy(outValue, entry.value.data(), copySize);
    }
    return 1;
  }

  // 2. Try wildcard and hierarchical matching
  // Collect all matching entries and select the best one based on:
  //   - Specificity (more specific patterns win)
  //   - Set time (last-set-wins for equal specificity)
  const ConfigDbEntry *bestMatch = nullptr;
  uint64_t bestSetTime = 0;
  size_t bestSpecificity = 0;  // Length of non-wildcard prefix

  for (const auto &[key, entry] : __moore_config_db_storage) {
    // Field name must match exactly
    if (entry.fieldName != lookupField)
      continue;

    // Check if this entry's instance path can match our lookup path
    if (!pathMatches(entry.instName, lookupInst))
      continue;

    // Calculate specificity: entries without wildcards are more specific
    // Among wildcard entries, longer non-wildcard prefixes are more specific
    size_t specificity = 0;
    if (!entry.hasWildcard) {
      // Exact prefix match (hierarchical) - very specific
      specificity = entry.instName.size() + 1000;  // Boost for non-wildcard
    } else {
      // Count characters before first wildcard
      size_t wildcardPos = entry.instName.find_first_of("*?");
      specificity = (wildcardPos != std::string::npos) ? wildcardPos : entry.instName.size();
    }

    // Select best match based on specificity, then set time
    if (bestMatch == nullptr ||
        specificity > bestSpecificity ||
        (specificity == bestSpecificity && entry.setTime > bestSetTime)) {
      bestMatch = &entry;
      bestSpecificity = specificity;
      bestSetTime = entry.setTime;
    }
  }

  if (bestMatch) {
    // Copy the value to the output buffer
    if (outValue && valueSize > 0 && !bestMatch->value.empty()) {
      size_t copySize = std::min(static_cast<size_t>(valueSize), bestMatch->value.size());
      std::memcpy(outValue, bestMatch->value.data(), copySize);
    }
    return 1;
  }

  // No match found
  return 0;
}

/// Check if a key exists in the configuration database.
/// This uses the same matching logic as __moore_config_db_get, so it returns
/// true if a get() would succeed (including wildcard/hierarchical matches).
/// @param instName Pointer to the instance name string data
/// @param instLen Length of the instance name string
/// @param fieldName Pointer to the field name string data
/// @param fieldLen Length of the field name string
/// @return 1 if the key exists (exact or via pattern), 0 otherwise
extern "C" int32_t __moore_config_db_exists(const char *instName, int64_t instLen,
                                            const char *fieldName, int64_t fieldLen) {
  auto [lookupInst, lookupField] = extractNames(instName, instLen, fieldName, fieldLen);
  std::string exactKey = buildConfigDbKey(instName, instLen, fieldName, fieldLen);

  std::lock_guard<std::mutex> lock(__moore_config_db_mutex);

  // Check exact match first
  if (__moore_config_db_storage.find(exactKey) != __moore_config_db_storage.end()) {
    return 1;
  }

  // Check for wildcard/hierarchical matches
  for (const auto &[key, entry] : __moore_config_db_storage) {
    if (entry.fieldName == lookupField && pathMatches(entry.instName, lookupInst)) {
      return 1;
    }
  }

  return 0;
}

/// Clear all entries from the configuration database.
/// This is useful for test cleanup between test cases.
extern "C" void __moore_config_db_clear(void) {
  std::lock_guard<std::mutex> lock(__moore_config_db_mutex);
  __moore_config_db_storage.clear();
  __moore_config_db_set_counter.store(0);
}

//===----------------------------------------------------------------------===//
// UVM Virtual Interface Binding Runtime
//===----------------------------------------------------------------------===//

namespace {

/// Information about a signal within an interface type.
struct VifSignalInfo {
  std::string signalName;  ///< Name of the signal
  int64_t offset;          ///< Byte offset within the interface instance
  int64_t size;            ///< Size of the signal in bytes
};

/// Information about a registered interface type.
struct VifInterfaceTypeInfo {
  std::string typeName;                             ///< Interface type name
  std::map<std::string, VifSignalInfo> signals;     ///< Signals in this interface
};

/// Internal representation of a virtual interface handle.
struct VifHandle {
  std::string interfaceTypeName;  ///< Name of the interface type
  std::string modportName;        ///< Optional modport name
  void *boundInstance;            ///< Pointer to the bound interface instance
  bool isBound;                   ///< Whether this vif is bound to an instance
};

/// Global registry of interface types and their signals.
std::map<std::string, VifInterfaceTypeInfo> __moore_vif_type_registry;
std::mutex __moore_vif_type_registry_mutex;

/// Global storage of all virtual interface handles.
std::vector<VifHandle *> __moore_vif_handles;
std::mutex __moore_vif_handles_mutex;

/// Helper to convert string parameters to std::string.
std::string vifMakeString(const char *data, int64_t len) {
  if (!data || len <= 0)
    return "";
  return std::string(data, static_cast<size_t>(len));
}

/// Look up signal info for an interface type.
const VifSignalInfo *lookupSignalInfo(const std::string &typeName,
                                      const std::string &signalName) {
  std::lock_guard<std::mutex> lock(__moore_vif_type_registry_mutex);
  auto typeIt = __moore_vif_type_registry.find(typeName);
  if (typeIt == __moore_vif_type_registry.end())
    return nullptr;
  auto sigIt = typeIt->second.signals.find(signalName);
  if (sigIt == typeIt->second.signals.end())
    return nullptr;
  return &sigIt->second;
}

} // namespace

extern "C" MooreVifHandle __moore_vif_create(const char *interfaceTypeName,
                                             int64_t interfaceTypeNameLen,
                                             const char *modportName,
                                             int64_t modportNameLen) {
  if (!interfaceTypeName || interfaceTypeNameLen <= 0)
    return MOORE_VIF_NULL;

  VifHandle *handle = new VifHandle();
  handle->interfaceTypeName = vifMakeString(interfaceTypeName, interfaceTypeNameLen);
  handle->modportName = vifMakeString(modportName, modportNameLen);
  handle->boundInstance = nullptr;
  handle->isBound = false;

  // Register the handle for tracking
  {
    std::lock_guard<std::mutex> lock(__moore_vif_handles_mutex);
    __moore_vif_handles.push_back(handle);
  }

  return static_cast<MooreVifHandle>(handle);
}

extern "C" int32_t __moore_vif_bind(MooreVifHandle vif, void *interfaceInstance) {
  if (!vif)
    return 0;

  VifHandle *handle = static_cast<VifHandle *>(vif);
  handle->boundInstance = interfaceInstance;
  handle->isBound = (interfaceInstance != nullptr);
  return 1;
}

extern "C" int32_t __moore_vif_is_bound(MooreVifHandle vif) {
  if (!vif)
    return 0;
  VifHandle *handle = static_cast<VifHandle *>(vif);
  return handle->isBound ? 1 : 0;
}

extern "C" void *__moore_vif_get_instance(MooreVifHandle vif) {
  if (!vif)
    return nullptr;
  VifHandle *handle = static_cast<VifHandle *>(vif);
  return handle->boundInstance;
}

extern "C" int32_t __moore_vif_get_signal(MooreVifHandle vif,
                                          const char *signalName,
                                          int64_t signalNameLen,
                                          void *outValue,
                                          int64_t valueSize) {
  if (!vif || !signalName || signalNameLen <= 0 || !outValue || valueSize <= 0)
    return 0;

  VifHandle *handle = static_cast<VifHandle *>(vif);
  if (!handle->isBound || !handle->boundInstance)
    return 0;

  std::string sigName = vifMakeString(signalName, signalNameLen);
  const VifSignalInfo *info = lookupSignalInfo(handle->interfaceTypeName, sigName);

  if (!info)
    return 0;

  // Calculate the pointer to the signal within the interface instance
  char *base = static_cast<char *>(handle->boundInstance);
  void *signalPtr = base + info->offset;

  // Copy the signal value to the output buffer
  int64_t copySize = std::min(valueSize, info->size);
  std::memcpy(outValue, signalPtr, static_cast<size_t>(copySize));

  return 1;
}

extern "C" int32_t __moore_vif_set_signal(MooreVifHandle vif,
                                          const char *signalName,
                                          int64_t signalNameLen,
                                          const void *value,
                                          int64_t valueSize) {
  if (!vif || !signalName || signalNameLen <= 0 || !value || valueSize <= 0)
    return 0;

  VifHandle *handle = static_cast<VifHandle *>(vif);
  if (!handle->isBound || !handle->boundInstance)
    return 0;

  std::string sigName = vifMakeString(signalName, signalNameLen);
  const VifSignalInfo *info = lookupSignalInfo(handle->interfaceTypeName, sigName);

  if (!info)
    return 0;

  // Calculate the pointer to the signal within the interface instance
  char *base = static_cast<char *>(handle->boundInstance);
  void *signalPtr = base + info->offset;

  // Copy the value to the signal storage
  int64_t copySize = std::min(valueSize, info->size);
  std::memcpy(signalPtr, value, static_cast<size_t>(copySize));

  return 1;
}

extern "C" void *__moore_vif_get_signal_ref(MooreVifHandle vif,
                                            const char *signalName,
                                            int64_t signalNameLen) {
  if (!vif || !signalName || signalNameLen <= 0)
    return nullptr;

  VifHandle *handle = static_cast<VifHandle *>(vif);
  if (!handle->isBound || !handle->boundInstance)
    return nullptr;

  std::string sigName = vifMakeString(signalName, signalNameLen);
  const VifSignalInfo *info = lookupSignalInfo(handle->interfaceTypeName, sigName);

  if (!info)
    return nullptr;

  // Calculate and return the pointer to the signal
  char *base = static_cast<char *>(handle->boundInstance);
  return base + info->offset;
}

extern "C" MooreString __moore_vif_get_type_name(MooreVifHandle vif) {
  MooreString result = {nullptr, 0};
  if (!vif)
    return result;

  VifHandle *handle = static_cast<VifHandle *>(vif);
  if (handle->interfaceTypeName.empty())
    return result;

  // Allocate and copy the type name
  result.len = static_cast<int64_t>(handle->interfaceTypeName.size());
  result.data = static_cast<char *>(std::malloc(static_cast<size_t>(result.len)));
  if (result.data) {
    std::memcpy(result.data, handle->interfaceTypeName.data(),
                static_cast<size_t>(result.len));
  } else {
    result.len = 0;
  }

  return result;
}

extern "C" MooreString __moore_vif_get_modport_name(MooreVifHandle vif) {
  MooreString result = {nullptr, 0};
  if (!vif)
    return result;

  VifHandle *handle = static_cast<VifHandle *>(vif);
  if (handle->modportName.empty())
    return result;

  // Allocate and copy the modport name
  result.len = static_cast<int64_t>(handle->modportName.size());
  result.data = static_cast<char *>(std::malloc(static_cast<size_t>(result.len)));
  if (result.data) {
    std::memcpy(result.data, handle->modportName.data(),
                static_cast<size_t>(result.len));
  } else {
    result.len = 0;
  }

  return result;
}

extern "C" int32_t __moore_vif_compare(MooreVifHandle vif1, MooreVifHandle vif2) {
  // Both null/unbound is considered equal
  if (!vif1 && !vif2)
    return 1;

  // One null, one not - not equal
  if (!vif1 || !vif2)
    return 0;

  VifHandle *h1 = static_cast<VifHandle *>(vif1);
  VifHandle *h2 = static_cast<VifHandle *>(vif2);

  // Both unbound is considered equal
  if (!h1->isBound && !h2->isBound)
    return 1;

  // One bound, one unbound - not equal
  if (h1->isBound != h2->isBound)
    return 0;

  // Both bound - compare the instance pointers
  return (h1->boundInstance == h2->boundInstance) ? 1 : 0;
}

extern "C" void __moore_vif_release(MooreVifHandle vif) {
  if (!vif)
    return;

  VifHandle *handle = static_cast<VifHandle *>(vif);

  // Remove from the global tracking list
  {
    std::lock_guard<std::mutex> lock(__moore_vif_handles_mutex);
    auto it = std::find(__moore_vif_handles.begin(), __moore_vif_handles.end(),
                        handle);
    if (it != __moore_vif_handles.end()) {
      __moore_vif_handles.erase(it);
    }
  }

  delete handle;
}

extern "C" void __moore_vif_clear_all(void) {
  std::lock_guard<std::mutex> lock(__moore_vif_handles_mutex);

  // Delete all handles
  for (VifHandle *handle : __moore_vif_handles) {
    delete handle;
  }
  __moore_vif_handles.clear();
}

extern "C" int32_t __moore_vif_register_signal(const char *interfaceTypeName,
                                               int64_t interfaceTypeNameLen,
                                               const char *signalName,
                                               int64_t signalNameLen,
                                               int64_t signalOffset,
                                               int64_t signalSize) {
  if (!interfaceTypeName || interfaceTypeNameLen <= 0 ||
      !signalName || signalNameLen <= 0 || signalSize <= 0)
    return 0;

  std::string typeName = vifMakeString(interfaceTypeName, interfaceTypeNameLen);
  std::string sigName = vifMakeString(signalName, signalNameLen);

  std::lock_guard<std::mutex> lock(__moore_vif_type_registry_mutex);

  // Create the interface type entry if it doesn't exist
  VifInterfaceTypeInfo &typeInfo = __moore_vif_type_registry[typeName];
  typeInfo.typeName = typeName;

  // Add or update the signal info
  VifSignalInfo &sigInfo = typeInfo.signals[sigName];
  sigInfo.signalName = sigName;
  sigInfo.offset = signalOffset;
  sigInfo.size = signalSize;

  return 1;
}

extern "C" void __moore_vif_clear_registry(void) {
  std::lock_guard<std::mutex> lock(__moore_vif_type_registry_mutex);
  __moore_vif_type_registry.clear();
}

//===----------------------------------------------------------------------===//
// UVM Component Hierarchy Support
//===----------------------------------------------------------------------===//

/// Get the full hierarchical name of a UVM component.
/// This function iteratively walks the parent chain to build the full name,
/// avoiding the recursion that cannot be inlined in LLHD IR.
///
/// Implementation:
/// 1. Collect all components in the hierarchy (from current to root)
/// 2. Build the full name by concatenating names from root to current
///
/// @param component Pointer to the component instance
/// @param parentOffset Byte offset of the m_parent field within the component
/// @param nameOffset Byte offset of the m_name field (MooreString) within the component
/// @return A new MooreString containing the full hierarchical name
extern "C" MooreString __moore_component_get_full_name(void *component,
                                                        int64_t parentOffset,
                                                        int64_t nameOffset) {
  // Handle null component
  if (!component) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  // Collect all components from current to root (we'll reverse later)
  std::vector<void *> hierarchy;
  void *current = component;

  while (current != nullptr) {
    hierarchy.push_back(current);

    // Get the parent pointer from the current component
    // The parent pointer is at offset parentOffset from the component base
    char *componentBytes = static_cast<char *>(current);
    void **parentPtr = reinterpret_cast<void **>(componentBytes + parentOffset);
    current = *parentPtr;
  }

  // Now build the full name from root to leaf
  // We iterate in reverse order (from root to current component)
  std::string fullName;

  for (auto it = hierarchy.rbegin(); it != hierarchy.rend(); ++it) {
    void *comp = *it;
    char *compBytes = static_cast<char *>(comp);

    // Get the name string from this component
    // The name is a MooreString at offset nameOffset
    MooreString *namePtr = reinterpret_cast<MooreString *>(compBytes + nameOffset);

    // Check if this component has a valid, non-empty name
    if (namePtr && namePtr->data && namePtr->len > 0) {
      // Skip empty names (like the root's name which is often empty)
      // Only add separator if we already have content
      if (!fullName.empty()) {
        fullName += ".";
      }
      fullName.append(namePtr->data, namePtr->len);
    }
  }

  // Allocate and return the result
  if (fullName.empty()) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  MooreString result = allocateString(fullName.length());
  std::memcpy(result.data, fullName.c_str(), fullName.length());
  return result;
}

//===----------------------------------------------------------------------===//
// UVM Runtime Infrastructure
//===----------------------------------------------------------------------===//
//
// These functions provide the basic UVM runtime support needed to execute
// UVM testbenches. The implementation starts simple and can be expanded to
// support more complex UVM features like the phase system and factory.
//

/// UVM phase start notification.
/// This function is called at the beginning of each UVM phase.
extern "C" void __uvm_phase_start(const char *phaseNameData,
                                   int64_t phaseNameLen) {
  std::string phaseName;
  if (phaseNameData && phaseNameLen > 0) {
    phaseName.assign(phaseNameData, static_cast<size_t>(phaseNameLen));
  }

  // Print UVM-style phase start message
  std::printf("UVM_INFO @ 0: uvm_test_top [PHASE] Starting %s_phase...\n",
              phaseName.empty() ? "(unknown)" : phaseName.c_str());
}

/// UVM phase end notification.
/// This function is called at the end of each UVM phase.
extern "C" void __uvm_phase_end(const char *phaseNameData,
                                 int64_t phaseNameLen) {
  std::string phaseName;
  if (phaseNameData && phaseNameLen > 0) {
    phaseName.assign(phaseNameData, static_cast<size_t>(phaseNameLen));
  }

  // Print UVM-style phase end message
  std::printf("UVM_INFO @ 0: uvm_test_top [PHASE] Completed %s_phase.\n",
              phaseName.empty() ? "(unknown)" : phaseName.c_str());
}

//===----------------------------------------------------------------------===//
// UVM Component Phase Callback Registration (Internal)
//===----------------------------------------------------------------------===//
//
// This section implements the component phase callback system that allows
// UVM components to register their phase methods with the runtime.
// Defined here so it can be used by __uvm_execute_phases below.
//

namespace {

/// Information about a registered UVM component.
struct UvmComponentInfo {
  void *component;                     // Pointer to the component instance
  std::string name;                    // Component instance name
  void *parent;                        // Parent component pointer
  int32_t depth;                       // Hierarchy depth (0 = root)
  int64_t handle;                      // Unique handle for this registration

  // Phase callbacks (one per phase)
  MooreUvmPhaseCallback phaseCallbacks[UVM_PHASE_COUNT];
  void *phaseUserData[UVM_PHASE_COUNT];

  // Task phase callback (for run_phase)
  MooreUvmTaskPhaseCallback runPhaseCallback;
  void *runPhaseUserData;

  UvmComponentInfo()
      : component(nullptr), parent(nullptr), depth(0), handle(0),
        runPhaseCallback(nullptr), runPhaseUserData(nullptr) {
    for (int i = 0; i < UVM_PHASE_COUNT; ++i) {
      phaseCallbacks[i] = nullptr;
      phaseUserData[i] = nullptr;
    }
  }
};

/// Global registry of UVM components.
struct UvmComponentRegistry {
  std::vector<UvmComponentInfo> components;
  int64_t nextHandle = 1;

  // Global phase callbacks
  void (*globalPhaseStartCallback)(MooreUvmPhase, const char *, void *) =
      nullptr;
  void *globalPhaseStartUserData = nullptr;
  void (*globalPhaseEndCallback)(MooreUvmPhase, const char *, void *) = nullptr;
  void *globalPhaseEndUserData = nullptr;

  /// Find a component by handle.
  UvmComponentInfo *findByHandle(int64_t handle) {
    for (auto &comp : components) {
      if (comp.handle == handle)
        return &comp;
    }
    return nullptr;
  }

  /// Get components sorted by depth for top-down traversal.
  std::vector<UvmComponentInfo *> getTopDown() {
    std::vector<UvmComponentInfo *> result;
    for (auto &comp : components) {
      result.push_back(&comp);
    }
    // Sort by depth ascending (parents before children)
    std::sort(result.begin(), result.end(),
              [](const UvmComponentInfo *a, const UvmComponentInfo *b) {
                return a->depth < b->depth;
              });
    return result;
  }

  /// Get components sorted by depth for bottom-up traversal.
  std::vector<UvmComponentInfo *> getBottomUp() {
    std::vector<UvmComponentInfo *> result;
    for (auto &comp : components) {
      result.push_back(&comp);
    }
    // Sort by depth descending (children before parents)
    std::sort(result.begin(), result.end(),
              [](const UvmComponentInfo *a, const UvmComponentInfo *b) {
                return a->depth > b->depth;
              });
    return result;
  }

  void clear() {
    components.clear();
    nextHandle = 1;
    globalPhaseStartCallback = nullptr;
    globalPhaseStartUserData = nullptr;
    globalPhaseEndCallback = nullptr;
    globalPhaseEndUserData = nullptr;
  }
};

/// Global component registry instance.
static UvmComponentRegistry &getComponentRegistry() {
  static UvmComponentRegistry registry;
  return registry;
}

/// Phase name lookup table for callback system.
static const char *callbackPhaseNames[UVM_PHASE_COUNT] = {
    "build",              "connect",  "end_of_elaboration",
    "start_of_simulation", "run",      "extract",
    "check",              "report",   "final"};

/// Check if a phase is top-down (build, final) or bottom-up.
static bool isTopDownPhase(MooreUvmPhase phase) {
  return phase == UVM_PHASE_BUILD || phase == UVM_PHASE_FINAL;
}

/// Execute callbacks for a specific phase.
static void executePhaseCallbacks(MooreUvmPhase phase) {
  auto &registry = getComponentRegistry();

  // Get the phase name
  const char *phaseName =
      (phase >= 0 && phase < UVM_PHASE_COUNT) ? callbackPhaseNames[phase] : "unknown";

  // Call global phase start callback
  if (registry.globalPhaseStartCallback) {
    registry.globalPhaseStartCallback(phase, phaseName,
                                       registry.globalPhaseStartUserData);
  }

  // Get components in the appropriate order
  std::vector<UvmComponentInfo *> orderedComponents =
      isTopDownPhase(phase) ? registry.getTopDown() : registry.getBottomUp();

  // Execute callbacks for each component
  for (auto *comp : orderedComponents) {
    if (phase == UVM_PHASE_RUN) {
      // For run_phase, use the task phase callback
      if (comp->runPhaseCallback) {
        comp->runPhaseCallback(comp->component, nullptr, comp->runPhaseUserData);
      }
    } else {
      // For function phases, use the regular callback
      if (comp->phaseCallbacks[phase]) {
        comp->phaseCallbacks[phase](comp->component, nullptr,
                                    comp->phaseUserData[phase]);
      }
    }
  }

  // Call global phase end callback
  if (registry.globalPhaseEndCallback) {
    registry.globalPhaseEndCallback(phase, phaseName,
                                     registry.globalPhaseEndUserData);
  }
}

} // anonymous namespace

/// Standard UVM phases in execution order.
static const char *uvmPhases[] = {
    "build",              // top-down: create component hierarchy
    "connect",            // bottom-up: connect TLM ports
    "end_of_elaboration", // bottom-up: fine-tune testbench
    "start_of_simulation", // bottom-up: get ready for simulation
    "run",                // task phase: main test execution
    "extract",            // bottom-up: extract data from DUT
    "check",              // bottom-up: check DUT state
    "report",             // bottom-up: report results
    "final"               // top-down: finalize simulation
};
static const size_t numUvmPhases = sizeof(uvmPhases) / sizeof(uvmPhases[0]);

/// Map phase index to MooreUvmPhase enum.
static MooreUvmPhase phaseIndexToEnum(size_t index) {
  static const MooreUvmPhase mapping[] = {
      UVM_PHASE_BUILD,
      UVM_PHASE_CONNECT,
      UVM_PHASE_END_OF_ELABORATION,
      UVM_PHASE_START_OF_SIMULATION,
      UVM_PHASE_RUN,
      UVM_PHASE_EXTRACT,
      UVM_PHASE_CHECK,
      UVM_PHASE_REPORT,
      UVM_PHASE_FINAL};
  if (index < sizeof(mapping) / sizeof(mapping[0]))
    return mapping[index];
  return UVM_PHASE_BUILD;
}

/// UVM phase execution.
/// Execute all standard UVM phases in sequence.
extern "C" void __uvm_execute_phases(void) {
  for (size_t i = 0; i < numUvmPhases; ++i) {
    const char *phase = uvmPhases[i];
    int64_t phaseLen = static_cast<int64_t>(std::strlen(phase));

    // Signal phase start
    __uvm_phase_start(phase, phaseLen);

    // Execute phase callbacks on all registered components.
    // The executePhaseCallbacks function handles:
    // - Top-down phases (build, final): traverse component tree top to bottom
    // - Bottom-up phases: traverse component tree bottom to top
    // - Task phases (run): currently executed synchronously (future: fork/join)
    MooreUvmPhase phaseEnum = phaseIndexToEnum(i);
    executePhaseCallbacks(phaseEnum);

    // Signal phase end
    __uvm_phase_end(phase, phaseLen);
  }
}

/// UVM run_test() implementation.
/// This is the main entry point for running UVM tests. It is called from
/// SystemVerilog code when run_test() is invoked.
///
/// @param testNameData Pointer to the test name string data
/// @param testNameLen Length of the test name string
///
/// This function:
/// 1. Creates the test component using the UVM factory (TODO)
/// 2. Executes the UVM phase sequence (build, connect, run, etc.)
/// 3. Reports summarize and finishes simulation
extern "C" void __uvm_run_test(const char *testNameData, int64_t testNameLen) {
  std::string testName;
  if (testNameData && testNameLen > 0) {
    testName.assign(testNameData, static_cast<size_t>(testNameLen));
  }

  // Print UVM-style message
  std::printf("UVM_INFO @ 0: uvm_test_top [RNTST] Running test %s...\n",
              testName.empty() ? "(default)" : testName.c_str());

  // Try to create the test component using the factory
  void *testComponent = nullptr;
  if (!testName.empty()) {
    // First check if the type is registered in the factory
    if (__moore_uvm_factory_is_type_registered(testName.c_str(),
                                               static_cast<int64_t>(testName.size()))) {
      // Create the test component with name "uvm_test_top"
      const char *instName = "uvm_test_top";
      testComponent = __moore_uvm_factory_create_component_by_name(
          testName.c_str(), static_cast<int64_t>(testName.size()), instName,
          static_cast<int64_t>(std::strlen(instName)), nullptr);

      if (testComponent) {
        std::printf(
            "UVM_INFO @ 0: uvm_test_top [RNTST] Test component '%s' created "
            "successfully.\n",
            testName.c_str());
      } else {
        std::printf(
            "UVM_ERROR @ 0: uvm_test_top [RNTST] Failed to create test '%s'.\n",
            testName.c_str());
      }
    } else {
      // Type not registered - print warning
      std::printf(
          "UVM_WARNING @ 0: uvm_test_top [NOTYPE] Test type '%s' is not "
          "registered with the factory. Test was not instantiated.\n",
          testName.c_str());
    }
  }

  // Execute all UVM phases
  __uvm_execute_phases();

  // Print completion message
  std::printf("UVM_INFO @ 0: uvm_test_top [FINISH] UVM phasing complete.\n");
}

//===----------------------------------------------------------------------===//
// UVM Factory Implementation
//===----------------------------------------------------------------------===//
//
// The factory maintains registries of component and object types that can be
// created by name. This enables late binding of test components and type/instance
// overrides as specified in IEEE 1800.2.
//

namespace {

/// Information about a registered component type.
struct UvmComponentTypeInfo {
  std::string typeName;
  MooreUvmComponentCreator creator;
  void *userData;
};

/// Information about a registered object type.
struct UvmObjectTypeInfo {
  std::string typeName;
  MooreUvmObjectCreator creator;
  void *userData;
};

/// The UVM factory registry.
struct UvmFactoryRegistry {
  std::map<std::string, UvmComponentTypeInfo> componentTypes;
  std::map<std::string, UvmObjectTypeInfo> objectTypes;
  std::map<std::string, std::string> typeOverrides;

  void clear() {
    componentTypes.clear();
    objectTypes.clear();
    typeOverrides.clear();
  }

  /// Apply type overrides to get the final type name.
  std::string resolveType(const std::string &typeName) const {
    std::string resolved = typeName;
    // Apply overrides (with loop detection)
    std::set<std::string> visited;
    while (typeOverrides.count(resolved) && !visited.count(resolved)) {
      visited.insert(resolved);
      resolved = typeOverrides.at(resolved);
    }
    return resolved;
  }
};

/// Global factory registry instance.
static UvmFactoryRegistry &getFactoryRegistry() {
  static UvmFactoryRegistry registry;
  return registry;
}

} // anonymous namespace

/// Register a component type with the factory.
extern "C" int32_t __moore_uvm_factory_register_component(
    const char *typeName, int64_t typeNameLen,
    MooreUvmComponentCreator creator, void *userData) {
  if (!typeName || typeNameLen <= 0 || !creator)
    return 0;

  auto &registry = getFactoryRegistry();
  std::string name(typeName, static_cast<size_t>(typeNameLen));

  // Check if already registered
  if (registry.componentTypes.count(name))
    return 0;

  UvmComponentTypeInfo info;
  info.typeName = name;
  info.creator = creator;
  info.userData = userData;
  registry.componentTypes[name] = info;

  return 1;
}

/// Register an object type with the factory.
extern "C" int32_t __moore_uvm_factory_register_object(
    const char *typeName, int64_t typeNameLen, MooreUvmObjectCreator creator,
    void *userData) {
  if (!typeName || typeNameLen <= 0 || !creator)
    return 0;

  auto &registry = getFactoryRegistry();
  std::string name(typeName, static_cast<size_t>(typeNameLen));

  // Check if already registered
  if (registry.objectTypes.count(name))
    return 0;

  UvmObjectTypeInfo info;
  info.typeName = name;
  info.creator = creator;
  info.userData = userData;
  registry.objectTypes[name] = info;

  return 1;
}

/// Create a component by type name.
extern "C" void *__moore_uvm_factory_create_component_by_name(
    const char *typeName, int64_t typeNameLen, const char *instName,
    int64_t instNameLen, void *parent) {
  if (!typeName || typeNameLen <= 0)
    return nullptr;

  auto &registry = getFactoryRegistry();
  std::string requestedType(typeName, static_cast<size_t>(typeNameLen));

  // Apply type overrides
  std::string resolvedType = registry.resolveType(requestedType);

  // Look up the type
  auto it = registry.componentTypes.find(resolvedType);
  if (it == registry.componentTypes.end()) {
    std::printf(
        "UVM_WARNING @ 0: uvm_factory [NOTYPE] Component type '%s' not found "
        "in factory registry.\n",
        resolvedType.c_str());
    return nullptr;
  }

  // Create the component
  return it->second.creator(instName, instNameLen, parent, it->second.userData);
}

/// Create an object by type name.
extern "C" void *__moore_uvm_factory_create_object_by_name(
    const char *typeName, int64_t typeNameLen, const char *instName,
    int64_t instNameLen) {
  if (!typeName || typeNameLen <= 0)
    return nullptr;

  auto &registry = getFactoryRegistry();
  std::string requestedType(typeName, static_cast<size_t>(typeNameLen));

  // Apply type overrides
  std::string resolvedType = registry.resolveType(requestedType);

  // Look up the type
  auto it = registry.objectTypes.find(resolvedType);
  if (it == registry.objectTypes.end()) {
    std::printf("UVM_WARNING @ 0: uvm_factory [NOTYPE] Object type '%s' not "
                "found in factory registry.\n",
                resolvedType.c_str());
    return nullptr;
  }

  // Create the object
  return it->second.creator(instName, instNameLen, it->second.userData);
}

/// Set a type override in the factory.
extern "C" int32_t __moore_uvm_factory_set_type_override(
    const char *originalType, int64_t originalTypeLen,
    const char *overrideType, int64_t overrideTypeLen, int32_t replace) {
  if (!originalType || originalTypeLen <= 0 || !overrideType ||
      overrideTypeLen <= 0)
    return 0;

  auto &registry = getFactoryRegistry();
  std::string original(originalType, static_cast<size_t>(originalTypeLen));
  std::string override(overrideType, static_cast<size_t>(overrideTypeLen));

  // Check if override already exists
  if (!replace && registry.typeOverrides.count(original))
    return 0;

  registry.typeOverrides[original] = override;
  return 1;
}

/// Check if a type is registered with the factory.
extern "C" int32_t __moore_uvm_factory_is_type_registered(const char *typeName,
                                                          int64_t typeNameLen) {
  if (!typeName || typeNameLen <= 0)
    return 0;

  auto &registry = getFactoryRegistry();
  std::string name(typeName, static_cast<size_t>(typeNameLen));

  return registry.componentTypes.count(name) ||
                 registry.objectTypes.count(name)
             ? 1
             : 0;
}

/// Get the number of registered types in the factory.
extern "C" int64_t __moore_uvm_factory_get_type_count(void) {
  auto &registry = getFactoryRegistry();
  return static_cast<int64_t>(registry.componentTypes.size() +
                              registry.objectTypes.size());
}

/// Clear all registered types and overrides from the factory.
extern "C" void __moore_uvm_factory_clear(void) {
  getFactoryRegistry().clear();
}

/// Print the factory state for debugging.
extern "C" void __moore_uvm_factory_print(void) {
  auto &registry = getFactoryRegistry();

  std::printf("UVM Factory State:\n");
  std::printf("  Registered component types: %zu\n",
              registry.componentTypes.size());
  for (const auto &[name, info] : registry.componentTypes) {
    std::printf("    - %s\n", name.c_str());
  }

  std::printf("  Registered object types: %zu\n", registry.objectTypes.size());
  for (const auto &[name, info] : registry.objectTypes) {
    std::printf("    - %s\n", name.c_str());
  }

  std::printf("  Type overrides: %zu\n", registry.typeOverrides.size());
  for (const auto &[original, override] : registry.typeOverrides) {
    std::printf("    - %s -> %s\n", original.c_str(), override.c_str());
  }
}

//===----------------------------------------------------------------------===//
// UVM Component Phase Callback Registration (Public API)
//===----------------------------------------------------------------------===//

/// Register a UVM component with the phase system.
extern "C" int64_t __moore_uvm_register_component(void *component,
                                                   const char *name,
                                                   int64_t nameLen,
                                                   void *parent,
                                                   int32_t depth) {
  if (!component)
    return 0;

  auto &registry = getComponentRegistry();

  UvmComponentInfo info;
  info.component = component;
  if (name && nameLen > 0) {
    info.name.assign(name, static_cast<size_t>(nameLen));
  }
  info.parent = parent;
  info.depth = depth;
  info.handle = registry.nextHandle++;

  registry.components.push_back(std::move(info));
  return registry.components.back().handle;
}

/// Unregister a UVM component from the phase system.
extern "C" void __moore_uvm_unregister_component(int64_t handle) {
  auto &registry = getComponentRegistry();

  auto it = std::remove_if(
      registry.components.begin(), registry.components.end(),
      [handle](const UvmComponentInfo &info) { return info.handle == handle; });

  registry.components.erase(it, registry.components.end());
}

/// Register a phase callback for a component.
extern "C" void __moore_uvm_set_phase_callback(int64_t handle,
                                                MooreUvmPhase phase,
                                                MooreUvmPhaseCallback callback,
                                                void *userData) {
  if (phase < 0 || phase >= UVM_PHASE_COUNT)
    return;

  auto &registry = getComponentRegistry();
  auto *comp = registry.findByHandle(handle);
  if (!comp)
    return;

  comp->phaseCallbacks[phase] = callback;
  comp->phaseUserData[phase] = userData;
}

/// Register a task phase callback for a component (for run_phase).
extern "C" void __moore_uvm_set_run_phase_callback(
    int64_t handle, MooreUvmTaskPhaseCallback callback, void *userData) {
  auto &registry = getComponentRegistry();
  auto *comp = registry.findByHandle(handle);
  if (!comp)
    return;

  comp->runPhaseCallback = callback;
  comp->runPhaseUserData = userData;
}

/// Get the number of registered components.
extern "C" int64_t __moore_uvm_get_component_count(void) {
  return static_cast<int64_t>(getComponentRegistry().components.size());
}

/// Clear all registered components and callbacks.
extern "C" void __moore_uvm_clear_components(void) {
  getComponentRegistry().clear();
}

/// Set a global phase start callback.
extern "C" void __moore_uvm_set_global_phase_start_callback(
    void (*callback)(MooreUvmPhase phase, const char *phaseName, void *userData),
    void *userData) {
  auto &registry = getComponentRegistry();
  registry.globalPhaseStartCallback = callback;
  registry.globalPhaseStartUserData = userData;
}

/// Set a global phase end callback.
extern "C" void __moore_uvm_set_global_phase_end_callback(
    void (*callback)(MooreUvmPhase phase, const char *phaseName, void *userData),
    void *userData) {
  auto &registry = getComponentRegistry();
  registry.globalPhaseEndCallback = callback;
  registry.globalPhaseEndUserData = userData;
}

//===----------------------------------------------------------------------===//
// TLM Port/Export Runtime Infrastructure
//===----------------------------------------------------------------------===//

namespace {

/// TLM tracing state
static bool tlmTraceEnabled = false;

/// TLM statistics
static int64_t tlmTotalConnections = 0;
static int64_t tlmTotalWrites = 0;
static int64_t tlmTotalGets = 0;

/// Forward declarations
struct TlmPort;
struct TlmFifo;

/// TLM port structure
struct TlmPort {
  std::string name;
  int64_t parentHandle;
  MooreTlmPortType type;
  std::vector<TlmPort *> connectedPorts;  // For analysis ports: list of subscribers
  TlmFifo *owningFifo;                    // If this is an analysis_export of a FIFO
  MooreTlmWriteCallback writeCallback;
  void *writeCallbackUserData;

  TlmPort(const std::string &n, int64_t parent, MooreTlmPortType t)
      : name(n), parentHandle(parent), type(t), owningFifo(nullptr),
        writeCallback(nullptr), writeCallbackUserData(nullptr) {}
};

/// TLM FIFO structure
struct TlmFifo {
  std::string name;
  int64_t parentHandle;
  int64_t maxSize;       // 0 = unbounded
  int64_t elementSize;
  std::vector<std::vector<uint8_t>> data;  // Queue of transactions
  TlmPort *analysisExport;  // The analysis_export port for this FIFO
  std::mutex mutex;  // For thread-safe operations
  std::condition_variable notEmpty;  // Condition variable for blocking get

  TlmFifo(const std::string &n, int64_t parent, int64_t max, int64_t elemSize)
      : name(n), parentHandle(parent), maxSize(max), elementSize(elemSize),
        analysisExport(nullptr) {}
};

/// Global TLM registry
struct TlmRegistry {
  std::vector<std::unique_ptr<TlmPort>> ports;
  std::vector<std::unique_ptr<TlmFifo>> fifos;
  std::mutex mutex;

  TlmPort *getPort(MooreTlmPortHandle handle) {
    std::lock_guard<std::mutex> lock(mutex);
    if (handle < 0 || static_cast<size_t>(handle) >= ports.size())
      return nullptr;
    return ports[handle].get();
  }

  TlmFifo *getFifo(MooreTlmFifoHandle handle) {
    std::lock_guard<std::mutex> lock(mutex);
    if (handle < 0 || static_cast<size_t>(handle) >= fifos.size())
      return nullptr;
    return fifos[handle].get();
  }

  MooreTlmPortHandle addPort(std::unique_ptr<TlmPort> port) {
    std::lock_guard<std::mutex> lock(mutex);
    MooreTlmPortHandle handle = static_cast<MooreTlmPortHandle>(ports.size());
    ports.push_back(std::move(port));
    return handle;
  }

  MooreTlmFifoHandle addFifo(std::unique_ptr<TlmFifo> fifo) {
    std::lock_guard<std::mutex> lock(mutex);
    MooreTlmFifoHandle handle = static_cast<MooreTlmFifoHandle>(fifos.size());
    fifos.push_back(std::move(fifo));
    return handle;
  }
};

TlmRegistry &getTlmRegistry() {
  static TlmRegistry registry;
  return registry;
}

void tlmTrace(const char *fmt, ...) {
  if (!tlmTraceEnabled)
    return;
  va_list args;
  va_start(args, fmt);
  std::printf("[TLM] ");
  std::vprintf(fmt, args);
  std::printf("\n");
  va_end(args);
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// TLM Port Operations
//===----------------------------------------------------------------------===//

extern "C" MooreTlmPortHandle __moore_tlm_port_create(const char *name,
                                                      int64_t nameLen,
                                                      int64_t parent,
                                                      MooreTlmPortType portType) {
  std::string portName(name, nameLen);
  auto port = std::make_unique<TlmPort>(portName, parent, portType);

  tlmTrace("Created port '%s' (type=%d)", portName.c_str(), portType);

  return getTlmRegistry().addPort(std::move(port));
}

extern "C" void __moore_tlm_port_destroy(MooreTlmPortHandle port) {
  auto *p = getTlmRegistry().getPort(port);
  if (p) {
    tlmTrace("Destroyed port '%s'", p->name.c_str());
    // Note: We don't actually remove from the vector to preserve handles
    // In a real implementation, we'd use a more sophisticated memory management
  }
}

extern "C" int32_t __moore_tlm_port_connect(MooreTlmPortHandle port,
                                            MooreTlmPortHandle export_) {
  auto *p = getTlmRegistry().getPort(port);
  auto *e = getTlmRegistry().getPort(export_);

  if (!p || !e) {
    std::fprintf(stderr, "[TLM] Error: Invalid port handles in connect()\n");
    return 0;
  }

  // For analysis ports, add the export to the subscriber list
  p->connectedPorts.push_back(e);
  tlmTotalConnections++;

  tlmTrace("Connected '%s' -> '%s'", p->name.c_str(), e->name.c_str());

  return 1;
}

extern "C" void __moore_tlm_port_write(MooreTlmPortHandle port,
                                       void *transaction,
                                       int64_t transactionSize) {
  auto *p = getTlmRegistry().getPort(port);
  if (!p) {
    std::fprintf(stderr, "[TLM] Error: Invalid port handle in write()\n");
    return;
  }

  tlmTrace("write() on port '%s' (size=%lld, subscribers=%zu)",
           p->name.c_str(), (long long)transactionSize, p->connectedPorts.size());

  tlmTotalWrites++;

  // Broadcast to all connected ports
  for (auto *subscriber : p->connectedPorts) {
    if (subscriber->writeCallback) {
      // Call the subscriber's write callback
      subscriber->writeCallback(subscriber->writeCallbackUserData,
                                transaction, transactionSize);
    } else if (subscriber->owningFifo) {
      // If connected to a FIFO's analysis_export, put the transaction in the FIFO
      TlmFifo *fifo = subscriber->owningFifo;
      std::lock_guard<std::mutex> lock(fifo->mutex);

      // Copy transaction data
      std::vector<uint8_t> txData(static_cast<uint8_t *>(transaction),
                                  static_cast<uint8_t *>(transaction) + transactionSize);
      fifo->data.push_back(std::move(txData));

      // Notify any waiting get() calls
      fifo->notEmpty.notify_one();

      tlmTrace("  -> Wrote to FIFO '%s' (size now %zu)",
               fifo->name.c_str(), fifo->data.size());
    }
  }
}

extern "C" MooreString __moore_tlm_port_get_name(MooreTlmPortHandle port) {
  auto *p = getTlmRegistry().getPort(port);
  if (!p) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  MooreString result;
  result.len = p->name.size();
  result.data = static_cast<char *>(std::malloc(result.len));
  if (result.data) {
    std::memcpy(result.data, p->name.c_str(), result.len);
  }
  return result;
}

extern "C" int64_t __moore_tlm_port_get_num_connections(MooreTlmPortHandle port) {
  auto *p = getTlmRegistry().getPort(port);
  if (!p)
    return 0;
  return static_cast<int64_t>(p->connectedPorts.size());
}

//===----------------------------------------------------------------------===//
// TLM FIFO Operations
//===----------------------------------------------------------------------===//

extern "C" MooreTlmFifoHandle __moore_tlm_fifo_create(const char *name,
                                                      int64_t nameLen,
                                                      int64_t parent,
                                                      int64_t maxSize,
                                                      int64_t elementSize) {
  std::string fifoName(name, nameLen);
  auto fifo = std::make_unique<TlmFifo>(fifoName, parent, maxSize, elementSize);

  // Create the analysis_export port for this FIFO
  std::string exportName = fifoName + ".analysis_export";
  auto exportPort = std::make_unique<TlmPort>(exportName, parent,
                                              MOORE_TLM_PORT_ANALYSIS);
  exportPort->owningFifo = fifo.get();

  MooreTlmPortHandle exportHandle = getTlmRegistry().addPort(std::move(exportPort));
  fifo->analysisExport = getTlmRegistry().getPort(exportHandle);

  tlmTrace("Created FIFO '%s' (maxSize=%lld, elementSize=%lld)",
           fifoName.c_str(), (long long)maxSize, (long long)elementSize);

  return getTlmRegistry().addFifo(std::move(fifo));
}

extern "C" void __moore_tlm_fifo_destroy(MooreTlmFifoHandle fifo) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (f) {
    tlmTrace("Destroyed FIFO '%s'", f->name.c_str());
  }
}

extern "C" MooreTlmPortHandle
__moore_tlm_fifo_get_analysis_export(MooreTlmFifoHandle fifo) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (!f || !f->analysisExport)
    return MOORE_TLM_INVALID_HANDLE;

  // Find the handle for the analysis_export port
  auto &registry = getTlmRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (size_t i = 0; i < registry.ports.size(); ++i) {
    if (registry.ports[i].get() == f->analysisExport) {
      return static_cast<MooreTlmPortHandle>(i);
    }
  }
  return MOORE_TLM_INVALID_HANDLE;
}

extern "C" int32_t __moore_tlm_fifo_try_put(MooreTlmFifoHandle fifo,
                                            void *transaction,
                                            int64_t transactionSize) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (!f)
    return 0;

  std::lock_guard<std::mutex> lock(f->mutex);

  // Check if bounded FIFO is full
  if (f->maxSize > 0 && static_cast<int64_t>(f->data.size()) >= f->maxSize)
    return 0;

  // Copy transaction data
  std::vector<uint8_t> txData(static_cast<uint8_t *>(transaction),
                              static_cast<uint8_t *>(transaction) + transactionSize);
  f->data.push_back(std::move(txData));

  // Notify any waiting get() calls
  f->notEmpty.notify_one();

  tlmTrace("try_put() on FIFO '%s' succeeded (size now %zu)",
           f->name.c_str(), f->data.size());

  return 1;
}

extern "C" void __moore_tlm_fifo_put(MooreTlmFifoHandle fifo,
                                     void *transaction,
                                     int64_t transactionSize) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (!f)
    return;

  std::unique_lock<std::mutex> lock(f->mutex);

  // For bounded FIFOs, wait until space is available
  // (For unbounded FIFOs, this is always immediate)
  // Note: In a real simulation environment, this would need proper
  // integration with the simulation scheduler

  // Copy transaction data
  std::vector<uint8_t> txData(static_cast<uint8_t *>(transaction),
                              static_cast<uint8_t *>(transaction) + transactionSize);
  f->data.push_back(std::move(txData));

  // Notify any waiting get() calls
  f->notEmpty.notify_one();

  tlmTrace("put() on FIFO '%s' (size now %zu)",
           f->name.c_str(), f->data.size());
}

extern "C" int32_t __moore_tlm_fifo_get(MooreTlmFifoHandle fifo,
                                        void *transaction,
                                        int64_t transactionSize) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (!f)
    return 0;

  std::unique_lock<std::mutex> lock(f->mutex);

  // Wait until data is available
  // Note: In a real simulation environment, this would need integration
  // with the simulation scheduler. For now, we use a condition variable
  // which works for threaded simulations.
  while (f->data.empty()) {
    // Use a timed wait to avoid infinite blocking in case of simulation issues
    auto status = f->notEmpty.wait_for(lock, std::chrono::milliseconds(100));
    if (status == std::cv_status::timeout && f->data.empty()) {
      // Keep waiting - in a real simulation, we'd yield to the scheduler here
      continue;
    }
  }

  // Get the front transaction
  auto &front = f->data.front();
  int64_t copySize = std::min(transactionSize, static_cast<int64_t>(front.size()));
  std::memcpy(transaction, front.data(), copySize);

  f->data.erase(f->data.begin());

  tlmTotalGets++;

  tlmTrace("get() on FIFO '%s' (size now %zu)",
           f->name.c_str(), f->data.size());

  return 1;
}

extern "C" int32_t __moore_tlm_fifo_try_get(MooreTlmFifoHandle fifo,
                                            void *transaction,
                                            int64_t transactionSize) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (!f)
    return 0;

  std::lock_guard<std::mutex> lock(f->mutex);

  if (f->data.empty())
    return 0;

  // Get the front transaction
  auto &front = f->data.front();
  int64_t copySize = std::min(transactionSize, static_cast<int64_t>(front.size()));
  std::memcpy(transaction, front.data(), copySize);

  f->data.erase(f->data.begin());

  tlmTotalGets++;

  tlmTrace("try_get() on FIFO '%s' succeeded (size now %zu)",
           f->name.c_str(), f->data.size());

  return 1;
}

extern "C" int32_t __moore_tlm_fifo_peek(MooreTlmFifoHandle fifo,
                                         void *transaction,
                                         int64_t transactionSize) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (!f)
    return 0;

  std::unique_lock<std::mutex> lock(f->mutex);

  // Wait until data is available
  while (f->data.empty()) {
    auto status = f->notEmpty.wait_for(lock, std::chrono::milliseconds(100));
    if (status == std::cv_status::timeout && f->data.empty()) {
      continue;
    }
  }

  // Peek at the front transaction (don't remove it)
  auto &front = f->data.front();
  int64_t copySize = std::min(transactionSize, static_cast<int64_t>(front.size()));
  std::memcpy(transaction, front.data(), copySize);

  tlmTrace("peek() on FIFO '%s'", f->name.c_str());

  return 1;
}

extern "C" int32_t __moore_tlm_fifo_try_peek(MooreTlmFifoHandle fifo,
                                             void *transaction,
                                             int64_t transactionSize) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (!f)
    return 0;

  std::lock_guard<std::mutex> lock(f->mutex);

  if (f->data.empty())
    return 0;

  // Peek at the front transaction (don't remove it)
  auto &front = f->data.front();
  int64_t copySize = std::min(transactionSize, static_cast<int64_t>(front.size()));
  std::memcpy(transaction, front.data(), copySize);

  tlmTrace("try_peek() on FIFO '%s' succeeded", f->name.c_str());

  return 1;
}

extern "C" int64_t __moore_tlm_fifo_size(MooreTlmFifoHandle fifo) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (!f)
    return 0;

  std::lock_guard<std::mutex> lock(f->mutex);
  return static_cast<int64_t>(f->data.size());
}

extern "C" int32_t __moore_tlm_fifo_is_empty(MooreTlmFifoHandle fifo) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (!f)
    return 1;

  std::lock_guard<std::mutex> lock(f->mutex);
  return f->data.empty() ? 1 : 0;
}

extern "C" int32_t __moore_tlm_fifo_is_full(MooreTlmFifoHandle fifo) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (!f)
    return 0;

  std::lock_guard<std::mutex> lock(f->mutex);

  // Unbounded FIFOs are never full
  if (f->maxSize == 0)
    return 0;

  return static_cast<int64_t>(f->data.size()) >= f->maxSize ? 1 : 0;
}

extern "C" void __moore_tlm_fifo_flush(MooreTlmFifoHandle fifo) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (!f)
    return;

  std::lock_guard<std::mutex> lock(f->mutex);
  f->data.clear();

  tlmTrace("flush() on FIFO '%s'", f->name.c_str());
}

extern "C" int32_t __moore_tlm_fifo_can_put(MooreTlmFifoHandle fifo) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (!f)
    return 0;

  std::lock_guard<std::mutex> lock(f->mutex);

  // Unbounded FIFOs can always accept puts
  if (f->maxSize == 0)
    return 1;

  // Check if there is free space
  return static_cast<int64_t>(f->data.size()) < f->maxSize ? 1 : 0;
}

extern "C" int32_t __moore_tlm_fifo_can_get(MooreTlmFifoHandle fifo) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (!f)
    return 0;

  std::lock_guard<std::mutex> lock(f->mutex);
  return f->data.empty() ? 0 : 1;
}

extern "C" int64_t __moore_tlm_fifo_used(MooreTlmFifoHandle fifo) {
  // Alias for size()
  return __moore_tlm_fifo_size(fifo);
}

extern "C" int64_t __moore_tlm_fifo_free(MooreTlmFifoHandle fifo) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (!f)
    return 0;

  std::lock_guard<std::mutex> lock(f->mutex);

  // Unbounded FIFOs have unlimited free space
  if (f->maxSize == 0)
    return INT64_MAX;

  // Return free slots
  return f->maxSize - static_cast<int64_t>(f->data.size());
}

extern "C" int64_t __moore_tlm_fifo_capacity(MooreTlmFifoHandle fifo) {
  auto *f = getTlmRegistry().getFifo(fifo);
  if (!f)
    return 0;

  return f->maxSize;
}

//===----------------------------------------------------------------------===//
// TLM Subscriber Operations
//===----------------------------------------------------------------------===//

extern "C" void __moore_tlm_subscriber_set_write_callback(
    MooreTlmPortHandle port, MooreTlmWriteCallback callback, void *userData) {
  auto *p = getTlmRegistry().getPort(port);
  if (!p)
    return;

  p->writeCallback = callback;
  p->writeCallbackUserData = userData;

  tlmTrace("Set write callback on port '%s'", p->name.c_str());
}

//===----------------------------------------------------------------------===//
// TLM Debugging/Tracing
//===----------------------------------------------------------------------===//

extern "C" void __moore_tlm_set_trace_enabled(int32_t enable) {
  tlmTraceEnabled = (enable != 0);
  if (tlmTraceEnabled) {
    std::printf("[TLM] Tracing enabled\n");
  }
}

extern "C" int32_t __moore_tlm_is_trace_enabled(void) {
  return tlmTraceEnabled ? 1 : 0;
}

extern "C" void __moore_tlm_print_topology(void) {
  auto &registry = getTlmRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);

  std::printf("\n=== TLM Connection Topology ===\n");

  std::printf("\nPorts (%zu):\n", registry.ports.size());
  for (size_t i = 0; i < registry.ports.size(); ++i) {
    auto &port = registry.ports[i];
    if (port) {
      std::printf("  [%zu] %s (type=%d, connections=%zu)\n",
                  i, port->name.c_str(), port->type, port->connectedPorts.size());
      for (auto *conn : port->connectedPorts) {
        std::printf("       -> %s\n", conn->name.c_str());
      }
    }
  }

  std::printf("\nFIFOs (%zu):\n", registry.fifos.size());
  for (size_t i = 0; i < registry.fifos.size(); ++i) {
    auto &fifo = registry.fifos[i];
    if (fifo) {
      std::printf("  [%zu] %s (maxSize=%lld, elementSize=%lld, current=%zu)\n",
                  i, fifo->name.c_str(), (long long)fifo->maxSize,
                  (long long)fifo->elementSize, fifo->data.size());
    }
  }

  std::printf("\n===============================\n");
}

extern "C" void __moore_tlm_get_statistics(int64_t *totalConnections,
                                           int64_t *totalWrites,
                                           int64_t *totalGets) {
  if (totalConnections)
    *totalConnections = tlmTotalConnections;
  if (totalWrites)
    *totalWrites = tlmTotalWrites;
  if (totalGets)
    *totalGets = tlmTotalGets;
}

//===----------------------------------------------------------------------===//
// UVM Objection System Implementation
//===----------------------------------------------------------------------===//

namespace {

/// Global flag to control objection tracing
static bool objectionTraceEnabled = false;

/// Objection entry tracking a single context's objections
struct ObjectionEntry {
  std::string context;      // Component path or context
  std::string description;  // Last description provided
  int64_t count;            // Number of objections from this context

  ObjectionEntry(const std::string &ctx, const std::string &desc, int64_t cnt)
      : context(ctx), description(desc), count(cnt) {}
};

/// Objection pool for a phase
struct ObjectionPool {
  std::string phaseName;                         // Name of the phase
  std::map<std::string, ObjectionEntry> entries; // Context -> entry map
  int64_t totalCount;                            // Total objections across all contexts
  int64_t drainTime;                             // Time to wait after zero
  std::mutex mutex;                              // For thread-safe operations
  std::condition_variable zeroCondition;         // Condition variable for wait_for_zero

  ObjectionPool(const std::string &name)
      : phaseName(name), totalCount(0), drainTime(0) {}
};

/// Global objection registry
struct ObjectionRegistry {
  std::vector<std::unique_ptr<ObjectionPool>> pools;
  std::mutex mutex;

  ObjectionPool *getPool(MooreObjectionHandle handle) {
    std::lock_guard<std::mutex> lock(mutex);
    if (handle < 0 || static_cast<size_t>(handle) >= pools.size())
      return nullptr;
    return pools[handle].get();
  }

  MooreObjectionHandle addPool(std::unique_ptr<ObjectionPool> pool) {
    std::lock_guard<std::mutex> lock(mutex);
    MooreObjectionHandle handle = static_cast<MooreObjectionHandle>(pools.size());
    pools.push_back(std::move(pool));
    return handle;
  }
};

ObjectionRegistry &getObjectionRegistry() {
  static ObjectionRegistry registry;
  return registry;
}

void objectionTrace(const char *fmt, ...) {
  if (!objectionTraceEnabled)
    return;
  va_list args;
  va_start(args, fmt);
  std::printf("[OBJECTION] ");
  std::vprintf(fmt, args);
  std::printf("\n");
  va_end(args);
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Objection System Operations
//===----------------------------------------------------------------------===//

extern "C" MooreObjectionHandle __moore_objection_create(const char *phaseName,
                                                          int64_t phaseNameLen) {
  std::string name(phaseName, phaseNameLen);
  auto pool = std::make_unique<ObjectionPool>(name);

  objectionTrace("Created objection pool for phase '%s'", name.c_str());

  return getObjectionRegistry().addPool(std::move(pool));
}

extern "C" void __moore_objection_destroy(MooreObjectionHandle objection) {
  auto *pool = getObjectionRegistry().getPool(objection);
  if (pool) {
    objectionTrace("Destroyed objection pool for phase '%s'",
                   pool->phaseName.c_str());
    // Note: We don't actually remove from the vector to preserve handles
  }
}

extern "C" void __moore_objection_raise(MooreObjectionHandle objection,
                                         const char *context, int64_t contextLen,
                                         const char *description,
                                         int64_t descriptionLen, int64_t count) {
  auto *pool = getObjectionRegistry().getPool(objection);
  if (!pool) {
    std::fprintf(stderr,
                 "[OBJECTION] Error: Invalid handle in raise_objection()\n");
    return;
  }

  std::string ctx = context ? std::string(context, contextLen) : "";
  std::string desc = description ? std::string(description, descriptionLen) : "";

  std::lock_guard<std::mutex> lock(pool->mutex);

  auto it = pool->entries.find(ctx);
  if (it != pool->entries.end()) {
    it->second.count += count;
    if (!desc.empty())
      it->second.description = desc;
  } else {
    pool->entries.emplace(ctx, ObjectionEntry(ctx, desc, count));
  }

  pool->totalCount += count;

  objectionTrace("raise_objection(phase='%s', context='%s', count=%lld) -> total=%lld",
                 pool->phaseName.c_str(), ctx.c_str(), (long long)count,
                 (long long)pool->totalCount);
}

extern "C" void __moore_objection_drop(MooreObjectionHandle objection,
                                        const char *context, int64_t contextLen,
                                        const char *description,
                                        int64_t descriptionLen, int64_t count) {
  auto *pool = getObjectionRegistry().getPool(objection);
  if (!pool) {
    std::fprintf(stderr,
                 "[OBJECTION] Error: Invalid handle in drop_objection()\n");
    return;
  }

  std::string ctx = context ? std::string(context, contextLen) : "";
  std::string desc = description ? std::string(description, descriptionLen) : "";

  std::unique_lock<std::mutex> lock(pool->mutex);

  auto it = pool->entries.find(ctx);
  if (it != pool->entries.end()) {
    it->second.count -= count;
    if (!desc.empty())
      it->second.description = desc;

    // Remove entry if count drops to zero or below
    if (it->second.count <= 0) {
      pool->entries.erase(it);
    }
  }

  pool->totalCount -= count;
  if (pool->totalCount < 0)
    pool->totalCount = 0;

  objectionTrace("drop_objection(phase='%s', context='%s', count=%lld) -> total=%lld",
                 pool->phaseName.c_str(), ctx.c_str(), (long long)count,
                 (long long)pool->totalCount);

  // Notify waiting threads if count reached zero
  if (pool->totalCount == 0) {
    lock.unlock();
    pool->zeroCondition.notify_all();
  }
}

extern "C" int64_t __moore_objection_get_count(MooreObjectionHandle objection) {
  auto *pool = getObjectionRegistry().getPool(objection);
  if (!pool)
    return 0;

  std::lock_guard<std::mutex> lock(pool->mutex);
  return pool->totalCount;
}

extern "C" int64_t
__moore_objection_get_count_by_context(MooreObjectionHandle objection,
                                        const char *context, int64_t contextLen) {
  auto *pool = getObjectionRegistry().getPool(objection);
  if (!pool)
    return 0;

  std::string ctx(context, contextLen);

  std::lock_guard<std::mutex> lock(pool->mutex);
  auto it = pool->entries.find(ctx);
  if (it != pool->entries.end())
    return it->second.count;
  return 0;
}

extern "C" void __moore_objection_set_drain_time(MooreObjectionHandle objection,
                                                  int64_t drainTime) {
  auto *pool = getObjectionRegistry().getPool(objection);
  if (!pool)
    return;

  std::lock_guard<std::mutex> lock(pool->mutex);
  pool->drainTime = drainTime;

  objectionTrace("set_drain_time(phase='%s', drainTime=%lld)",
                 pool->phaseName.c_str(), (long long)drainTime);
}

extern "C" int64_t
__moore_objection_get_drain_time(MooreObjectionHandle objection) {
  auto *pool = getObjectionRegistry().getPool(objection);
  if (!pool)
    return 0;

  std::lock_guard<std::mutex> lock(pool->mutex);
  return pool->drainTime;
}

extern "C" int32_t
__moore_objection_wait_for_zero(MooreObjectionHandle objection) {
  auto *pool = getObjectionRegistry().getPool(objection);
  if (!pool)
    return 0;

  std::unique_lock<std::mutex> lock(pool->mutex);

  // Wait until total count reaches zero
  pool->zeroCondition.wait(lock, [pool]() { return pool->totalCount == 0; });

  // Apply drain time if set
  // Note: In a real simulation environment, this would integrate with the
  // simulation scheduler. For now, we use std::this_thread::sleep_for
  // as a simple approximation.
  if (pool->drainTime > 0) {
    int64_t drainMs = pool->drainTime;
    lock.unlock();

    objectionTrace("wait_for_zero(phase='%s') - starting drain time: %lldms",
                   pool->phaseName.c_str(), (long long)drainMs);

    // Sleep for drain time (interpreting drainTime as milliseconds for testing)
    std::this_thread::sleep_for(std::chrono::milliseconds(drainMs));

    objectionTrace("wait_for_zero(phase='%s') - drain time complete",
                   pool->phaseName.c_str());
  } else {
    objectionTrace("wait_for_zero(phase='%s') - completed (no drain time)",
                   pool->phaseName.c_str());
  }

  return 1;
}

extern "C" int32_t __moore_objection_is_zero(MooreObjectionHandle objection) {
  auto *pool = getObjectionRegistry().getPool(objection);
  if (!pool)
    return 1; // Treat invalid handle as "zero" to avoid deadlocks

  std::lock_guard<std::mutex> lock(pool->mutex);
  return pool->totalCount == 0 ? 1 : 0;
}

extern "C" MooreString
__moore_objection_get_phase_name(MooreObjectionHandle objection) {
  auto *pool = getObjectionRegistry().getPool(objection);
  if (!pool) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  MooreString result;
  result.len = pool->phaseName.size();
  result.data = static_cast<char *>(std::malloc(result.len));
  if (result.data) {
    std::memcpy(result.data, pool->phaseName.c_str(), result.len);
  }
  return result;
}

extern "C" void __moore_objection_set_trace_enabled(int32_t enable) {
  objectionTraceEnabled = (enable != 0);
  if (objectionTraceEnabled) {
    std::printf("[OBJECTION] Tracing enabled\n");
  }
}

extern "C" int32_t __moore_objection_is_trace_enabled(void) {
  return objectionTraceEnabled ? 1 : 0;
}

extern "C" void __moore_objection_print_summary(void) {
  auto &registry = getObjectionRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);

  std::printf("\n=== UVM Objection Summary ===\n");

  if (registry.pools.empty()) {
    std::printf("No objection pools registered.\n");
  } else {
    for (size_t i = 0; i < registry.pools.size(); ++i) {
      auto &pool = registry.pools[i];
      if (pool) {
        std::lock_guard<std::mutex> poolLock(pool->mutex);
        std::printf("\n[%zu] Phase: %s\n", i, pool->phaseName.c_str());
        std::printf("     Total Count: %lld\n", (long long)pool->totalCount);
        std::printf("     Drain Time: %lld\n", (long long)pool->drainTime);

        if (!pool->entries.empty()) {
          std::printf("     Objections by context:\n");
          for (const auto &entry : pool->entries) {
            std::printf("       '%s': %lld", entry.first.c_str(),
                        (long long)entry.second.count);
            if (!entry.second.description.empty()) {
              std::printf(" (%s)", entry.second.description.c_str());
            }
            std::printf("\n");
          }
        }
      }
    }
  }

  std::printf("\n=============================\n");
}

//===----------------------------------------------------------------------===//
// UVM Sequence/Sequencer Infrastructure Implementation
//===----------------------------------------------------------------------===//

namespace {

/// Forward declarations
struct Sequence;
struct Sequencer;

/// Pending item waiting to be transferred to the driver
struct PendingItem {
  MooreSequenceHandle sequence;  ///< Sequence that owns the item
  std::vector<uint8_t> data;     ///< Item data
  bool itemReady;                ///< True when item is ready for driver
  bool itemDone;                 ///< True when driver has processed item
  std::vector<uint8_t> response; ///< Response data from driver

  PendingItem(MooreSequenceHandle seq, void *item, int64_t size)
      : sequence(seq), itemReady(false), itemDone(false) {
    if (item && size > 0) {
      data.resize(size);
      std::memcpy(data.data(), item, size);
    }
  }
};

/// Sequence state
struct Sequence {
  std::string name;
  int32_t priority;
  MooreSeqState state;
  MooreSequencerHandle parentSequencer;

  // Synchronization for start_item/finish_item
  std::mutex mutex;
  std::condition_variable cv;
  bool stopRequested;
  bool waitingForDriver;    ///< Waiting in start_item for driver
  bool waitingForItemDone;  ///< Waiting in finish_item for item_done
  bool driverReady;         ///< Driver is ready to accept item
  bool itemDoneSignaled;    ///< item_done has been called

  // Current item being transferred
  std::shared_ptr<PendingItem> currentItem;

  // For async execution
  std::thread executionThread;
  std::atomic<bool> asyncRunning{false};

  Sequence(const std::string &n, int32_t prio)
      : name(n), priority(prio), state(MOORE_SEQ_STATE_IDLE),
        parentSequencer(MOORE_SEQUENCER_INVALID_HANDLE), stopRequested(false),
        waitingForDriver(false), waitingForItemDone(false),
        driverReady(false), itemDoneSignaled(false) {}

  ~Sequence() {
    // Ensure thread is joined if running async
    if (executionThread.joinable()) {
      {
        std::lock_guard<std::mutex> lock(mutex);
        stopRequested = true;
      }
      cv.notify_all();
      executionThread.join();
    }
  }
};

/// Sequencer state
struct Sequencer {
  std::string name;
  int64_t parent;
  bool running;
  MooreSeqArbMode arbMode;

  // User-defined arbitration
  MooreSeqArbCallback arbCallback;
  void *arbUserData;

  // Synchronization
  std::mutex mutex;
  std::condition_variable cv;

  // Sequences waiting for driver access
  std::vector<MooreSequenceHandle> waitingSequences;

  // Current active sequence (the one whose item is being processed)
  MooreSequenceHandle activeSequence;
  bool hasActiveItem;
  std::shared_ptr<PendingItem> activeItem;

  Sequencer(const std::string &n, int64_t p)
      : name(n), parent(p), running(false), arbMode(MOORE_SEQ_ARB_FIFO),
        arbCallback(nullptr), arbUserData(nullptr),
        activeSequence(MOORE_SEQUENCE_INVALID_HANDLE), hasActiveItem(false) {}
};

/// Registry for all sequencers and sequences
struct SeqRegistry {
  std::vector<std::unique_ptr<Sequencer>> sequencers;
  std::vector<std::unique_ptr<Sequence>> sequences;
  std::mutex mutex;

  // Statistics
  std::atomic<int64_t> totalSequencesCreated{0};
  std::atomic<int64_t> totalItemsTransferred{0};
  std::atomic<int64_t> totalArbitrations{0};

  Sequencer *getSequencer(MooreSequencerHandle handle) {
    if (handle < 0 || static_cast<size_t>(handle) >= sequencers.size())
      return nullptr;
    return sequencers[handle].get();
  }

  Sequence *getSequence(MooreSequenceHandle handle) {
    if (handle < 0 || static_cast<size_t>(handle) >= sequences.size())
      return nullptr;
    return sequences[handle].get();
  }

  MooreSequencerHandle addSequencer(std::unique_ptr<Sequencer> sequencer) {
    std::lock_guard<std::mutex> lock(mutex);
    MooreSequencerHandle handle = static_cast<MooreSequencerHandle>(sequencers.size());
    sequencers.push_back(std::move(sequencer));
    return handle;
  }

  MooreSequenceHandle addSequence(std::unique_ptr<Sequence> sequence) {
    std::lock_guard<std::mutex> lock(mutex);
    MooreSequenceHandle handle = static_cast<MooreSequenceHandle>(sequences.size());
    sequences.push_back(std::move(sequence));
    totalSequencesCreated++;
    return handle;
  }
};

SeqRegistry &getSeqRegistry() {
  static SeqRegistry registry;
  return registry;
}

// Tracing flag
static std::atomic<bool> seqTraceEnabled{false};

/// Perform arbitration to select the next sequence
MooreSequenceHandle arbitrateSequence(Sequencer *sequencer) {
  if (sequencer->waitingSequences.empty())
    return MOORE_SEQUENCE_INVALID_HANDLE;

  auto &registry = getSeqRegistry();
  registry.totalArbitrations++;

  switch (sequencer->arbMode) {
    case MOORE_SEQ_ARB_FIFO:
    default: {
      // First-in, first-out
      return sequencer->waitingSequences.front();
    }

    case MOORE_SEQ_ARB_RANDOM: {
      // Random selection
      size_t idx = std::rand() % sequencer->waitingSequences.size();
      return sequencer->waitingSequences[idx];
    }

    case MOORE_SEQ_ARB_WEIGHTED: {
      // Weighted by priority
      std::vector<std::pair<MooreSequenceHandle, int32_t>> weighted;
      int32_t totalWeight = 0;
      for (auto h : sequencer->waitingSequences) {
        auto *seq = registry.getSequence(h);
        if (seq) {
          int32_t w = seq->priority > 0 ? seq->priority : 1;
          weighted.push_back({h, w});
          totalWeight += w;
        }
      }
      if (totalWeight <= 0 || weighted.empty())
        return sequencer->waitingSequences.front();

      int32_t r = std::rand() % totalWeight;
      int32_t cumulative = 0;
      for (auto &p : weighted) {
        cumulative += p.second;
        if (r < cumulative)
          return p.first;
      }
      return weighted.back().first;
    }

    case MOORE_SEQ_ARB_STRICT_FIFO: {
      // Highest priority first, FIFO among equals
      int32_t maxPriority = INT_MIN;
      MooreSequenceHandle result = MOORE_SEQUENCE_INVALID_HANDLE;
      for (auto h : sequencer->waitingSequences) {
        auto *seq = registry.getSequence(h);
        if (seq && seq->priority > maxPriority) {
          maxPriority = seq->priority;
          result = h;
        }
      }
      return result;
    }

    case MOORE_SEQ_ARB_STRICT_RANDOM: {
      // Random among highest priority sequences
      int32_t maxPriority = INT_MIN;
      for (auto h : sequencer->waitingSequences) {
        auto *seq = registry.getSequence(h);
        if (seq && seq->priority > maxPriority)
          maxPriority = seq->priority;
      }

      std::vector<MooreSequenceHandle> candidates;
      for (auto h : sequencer->waitingSequences) {
        auto *seq = registry.getSequence(h);
        if (seq && seq->priority == maxPriority)
          candidates.push_back(h);
      }

      if (candidates.empty())
        return MOORE_SEQUENCE_INVALID_HANDLE;
      return candidates[std::rand() % candidates.size()];
    }

    case MOORE_SEQ_ARB_USER: {
      // User-defined arbitration
      if (sequencer->arbCallback) {
        // Find the sequencer handle by searching the registry
        MooreSequencerHandle seqrHandle = MOORE_SEQUENCER_INVALID_HANDLE;
        auto &seqrs = registry.sequencers;
        for (size_t i = 0; i < seqrs.size(); ++i) {
          if (seqrs[i].get() == sequencer) {
            seqrHandle = static_cast<MooreSequencerHandle>(i);
            break;
          }
        }
        int32_t idx = sequencer->arbCallback(
            seqrHandle,
            sequencer->waitingSequences.data(),
            static_cast<int32_t>(sequencer->waitingSequences.size()),
            sequencer->arbUserData);
        if (idx >= 0 && static_cast<size_t>(idx) < sequencer->waitingSequences.size())
          return sequencer->waitingSequences[idx];
      }
      return sequencer->waitingSequences.front();
    }
  }
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Sequencer Operations
//===----------------------------------------------------------------------===//

extern "C" MooreSequencerHandle __moore_sequencer_create(const char *name,
                                                          int64_t nameLen,
                                                          int64_t parent) {
  std::string seqrName(name, nameLen);
  auto sequencer = std::make_unique<Sequencer>(seqrName, parent);

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Created sequencer '%s'\n", seqrName.c_str());
  }

  return getSeqRegistry().addSequencer(std::move(sequencer));
}

extern "C" void __moore_sequencer_destroy(MooreSequencerHandle sequencer) {
  auto *seqr = getSeqRegistry().getSequencer(sequencer);
  if (!seqr)
    return;

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Destroying sequencer '%s'\n", seqr->name.c_str());
  }

  // Stop the sequencer
  __moore_sequencer_stop(sequencer);

  // Note: We don't actually delete from vector to keep handles valid
  // In a production system, we'd use a more sophisticated handle system
}

extern "C" void __moore_sequencer_start(MooreSequencerHandle sequencer) {
  auto *seqr = getSeqRegistry().getSequencer(sequencer);
  if (!seqr)
    return;

  std::lock_guard<std::mutex> lock(seqr->mutex);
  seqr->running = true;

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Started sequencer '%s'\n", seqr->name.c_str());
  }
}

extern "C" void __moore_sequencer_stop(MooreSequencerHandle sequencer) {
  auto *seqr = getSeqRegistry().getSequencer(sequencer);
  if (!seqr)
    return;

  {
    std::lock_guard<std::mutex> lock(seqr->mutex);
    seqr->running = false;
  }
  seqr->cv.notify_all();

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Stopped sequencer '%s'\n", seqr->name.c_str());
  }
}

extern "C" int32_t __moore_sequencer_is_running(MooreSequencerHandle sequencer) {
  auto *seqr = getSeqRegistry().getSequencer(sequencer);
  if (!seqr)
    return 0;

  std::lock_guard<std::mutex> lock(seqr->mutex);
  return seqr->running ? 1 : 0;
}

extern "C" void __moore_sequencer_set_arbitration(MooreSequencerHandle sequencer,
                                                   MooreSeqArbMode mode) {
  auto *seqr = getSeqRegistry().getSequencer(sequencer);
  if (!seqr)
    return;

  std::lock_guard<std::mutex> lock(seqr->mutex);
  seqr->arbMode = mode;

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Sequencer '%s' arbitration set to %d\n",
                seqr->name.c_str(), mode);
  }
}

extern "C" MooreSeqArbMode
__moore_sequencer_get_arbitration(MooreSequencerHandle sequencer) {
  auto *seqr = getSeqRegistry().getSequencer(sequencer);
  if (!seqr)
    return MOORE_SEQ_ARB_FIFO;

  std::lock_guard<std::mutex> lock(seqr->mutex);
  return seqr->arbMode;
}

extern "C" void __moore_sequencer_set_arb_callback(MooreSequencerHandle sequencer,
                                                    MooreSeqArbCallback callback,
                                                    void *userData) {
  auto *seqr = getSeqRegistry().getSequencer(sequencer);
  if (!seqr)
    return;

  std::lock_guard<std::mutex> lock(seqr->mutex);
  seqr->arbCallback = callback;
  seqr->arbUserData = userData;
}

extern "C" MooreString __moore_sequencer_get_name(MooreSequencerHandle sequencer) {
  auto *seqr = getSeqRegistry().getSequencer(sequencer);
  if (!seqr) {
    MooreString result = {nullptr, 0};
    return result;
  }

  std::lock_guard<std::mutex> lock(seqr->mutex);
  MooreString result;
  result.len = static_cast<int64_t>(seqr->name.size());
  result.data = static_cast<char *>(std::malloc(result.len));
  if (result.data) {
    std::memcpy(result.data, seqr->name.data(), result.len);
  }
  return result;
}

extern "C" int32_t
__moore_sequencer_get_num_waiting(MooreSequencerHandle sequencer) {
  auto *seqr = getSeqRegistry().getSequencer(sequencer);
  if (!seqr)
    return 0;

  std::lock_guard<std::mutex> lock(seqr->mutex);
  return static_cast<int32_t>(seqr->waitingSequences.size());
}

//===----------------------------------------------------------------------===//
// Sequence Operations
//===----------------------------------------------------------------------===//

extern "C" MooreSequenceHandle __moore_sequence_create(const char *name,
                                                        int64_t nameLen,
                                                        int32_t priority) {
  std::string seqName(name, nameLen);
  auto sequence = std::make_unique<Sequence>(seqName, priority);

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Created sequence '%s' (priority=%d)\n",
                seqName.c_str(), priority);
  }

  return getSeqRegistry().addSequence(std::move(sequence));
}

extern "C" void __moore_sequence_destroy(MooreSequenceHandle sequence) {
  auto *seq = getSeqRegistry().getSequence(sequence);
  if (!seq)
    return;

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Destroying sequence '%s'\n", seq->name.c_str());
  }

  // Stop the sequence if running
  __moore_sequence_stop(sequence);
}

extern "C" int32_t __moore_sequence_start(MooreSequenceHandle sequence,
                                           MooreSequencerHandle sequencer,
                                           MooreSequenceBodyCallback body,
                                           void *userData) {
  auto &registry = getSeqRegistry();
  auto *seq = registry.getSequence(sequence);
  auto *seqr = registry.getSequencer(sequencer);

  if (!seq || !seqr || !body)
    return 0;

  {
    std::lock_guard<std::mutex> seqLock(seq->mutex);
    if (seq->state != MOORE_SEQ_STATE_IDLE &&
        seq->state != MOORE_SEQ_STATE_FINISHED &&
        seq->state != MOORE_SEQ_STATE_STOPPED)
      return 0;

    seq->state = MOORE_SEQ_STATE_RUNNING;
    seq->parentSequencer = sequencer;
    seq->stopRequested = false;
  }

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Starting sequence '%s' on sequencer '%s'\n",
                seq->name.c_str(), seqr->name.c_str());
  }

  // Execute the sequence body
  body(sequence, userData);

  {
    std::lock_guard<std::mutex> seqLock(seq->mutex);
    if (seq->stopRequested) {
      seq->state = MOORE_SEQ_STATE_STOPPED;
    } else {
      seq->state = MOORE_SEQ_STATE_FINISHED;
    }
  }

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Sequence '%s' completed\n", seq->name.c_str());
  }

  return seq->state == MOORE_SEQ_STATE_FINISHED ? 1 : 0;
}

extern "C" int32_t __moore_sequence_start_async(MooreSequenceHandle sequence,
                                                 MooreSequencerHandle sequencer,
                                                 MooreSequenceBodyCallback body,
                                                 void *userData) {
  auto &registry = getSeqRegistry();
  auto *seq = registry.getSequence(sequence);
  auto *seqr = registry.getSequencer(sequencer);

  if (!seq || !seqr || !body)
    return 0;

  {
    std::lock_guard<std::mutex> seqLock(seq->mutex);
    if (seq->asyncRunning.load())
      return 0;

    seq->state = MOORE_SEQ_STATE_RUNNING;
    seq->parentSequencer = sequencer;
    seq->stopRequested = false;
    seq->asyncRunning = true;
  }

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Starting async sequence '%s' on sequencer '%s'\n",
                seq->name.c_str(), seqr->name.c_str());
  }

  // Start the sequence in a background thread
  seq->executionThread = std::thread([sequence, body, userData]() {
    auto *seq = getSeqRegistry().getSequence(sequence);
    if (!seq)
      return;

    body(sequence, userData);

    {
      std::lock_guard<std::mutex> seqLock(seq->mutex);
      if (seq->stopRequested) {
        seq->state = MOORE_SEQ_STATE_STOPPED;
      } else {
        seq->state = MOORE_SEQ_STATE_FINISHED;
      }
      seq->asyncRunning = false;
    }
    seq->cv.notify_all();
  });

  return 1;
}

extern "C" int32_t __moore_sequence_wait(MooreSequenceHandle sequence) {
  auto *seq = getSeqRegistry().getSequence(sequence);
  if (!seq)
    return 0;

  if (seq->executionThread.joinable()) {
    seq->executionThread.join();
  }

  return seq->state == MOORE_SEQ_STATE_FINISHED ? 1 : 0;
}

extern "C" void __moore_sequence_stop(MooreSequenceHandle sequence) {
  auto *seq = getSeqRegistry().getSequence(sequence);
  if (!seq)
    return;

  {
    std::lock_guard<std::mutex> lock(seq->mutex);
    seq->stopRequested = true;
    seq->state = MOORE_SEQ_STATE_STOPPED;
  }
  seq->cv.notify_all();

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Stopping sequence '%s'\n", seq->name.c_str());
  }
}

extern "C" MooreSeqState __moore_sequence_get_state(MooreSequenceHandle sequence) {
  auto *seq = getSeqRegistry().getSequence(sequence);
  if (!seq)
    return MOORE_SEQ_STATE_IDLE;

  std::lock_guard<std::mutex> lock(seq->mutex);
  return seq->state;
}

extern "C" MooreString __moore_sequence_get_name(MooreSequenceHandle sequence) {
  auto *seq = getSeqRegistry().getSequence(sequence);
  if (!seq) {
    MooreString result = {nullptr, 0};
    return result;
  }

  std::lock_guard<std::mutex> lock(seq->mutex);
  MooreString result;
  result.len = static_cast<int64_t>(seq->name.size());
  result.data = static_cast<char *>(std::malloc(result.len));
  if (result.data) {
    std::memcpy(result.data, seq->name.data(), result.len);
  }
  return result;
}

extern "C" int32_t __moore_sequence_get_priority(MooreSequenceHandle sequence) {
  auto *seq = getSeqRegistry().getSequence(sequence);
  if (!seq)
    return 0;

  std::lock_guard<std::mutex> lock(seq->mutex);
  return seq->priority;
}

extern "C" void __moore_sequence_set_priority(MooreSequenceHandle sequence,
                                               int32_t priority) {
  auto *seq = getSeqRegistry().getSequence(sequence);
  if (!seq)
    return;

  std::lock_guard<std::mutex> lock(seq->mutex);
  seq->priority = priority;
}

//===----------------------------------------------------------------------===//
// Sequence-Driver Handshake
//===----------------------------------------------------------------------===//

extern "C" int32_t __moore_sequence_start_item(MooreSequenceHandle sequence,
                                                void *item, int64_t itemSize) {
  auto &registry = getSeqRegistry();
  auto *seq = registry.getSequence(sequence);
  if (!seq || !item || itemSize <= 0)
    return 0;

  MooreSequencerHandle seqrHandle;
  {
    std::lock_guard<std::mutex> lock(seq->mutex);
    if (seq->stopRequested)
      return 0;
    seqrHandle = seq->parentSequencer;
  }

  auto *seqr = registry.getSequencer(seqrHandle);
  if (!seqr)
    return 0;

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Sequence '%s' calling start_item\n",
                seq->name.c_str());
  }

  // Create the pending item
  auto pendingItem = std::make_shared<PendingItem>(sequence, item, itemSize);

  {
    std::lock_guard<std::mutex> seqLock(seq->mutex);
    seq->currentItem = pendingItem;
    seq->waitingForDriver = true;
    seq->driverReady = false;
    seq->state = MOORE_SEQ_STATE_WAITING;
  }

  // Add to sequencer's waiting queue
  {
    std::lock_guard<std::mutex> seqrLock(seqr->mutex);
    seqr->waitingSequences.push_back(sequence);
  }
  seqr->cv.notify_all();

  // Wait for driver to be ready (driver calls get_next_item)
  {
    std::unique_lock<std::mutex> lock(seq->mutex);
    seq->cv.wait(lock, [&seq]() {
      return seq->driverReady || seq->stopRequested;
    });

    if (seq->stopRequested) {
      seq->waitingForDriver = false;
      return 0;
    }

    seq->waitingForDriver = false;
    seq->state = MOORE_SEQ_STATE_RUNNING;
  }

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Sequence '%s' start_item granted\n",
                seq->name.c_str());
  }

  return 1;
}

extern "C" int32_t __moore_sequence_finish_item(MooreSequenceHandle sequence,
                                                 void *item, int64_t itemSize) {
  auto &registry = getSeqRegistry();
  auto *seq = registry.getSequence(sequence);
  if (!seq || !item || itemSize <= 0)
    return 0;

  MooreSequencerHandle seqrHandle;
  std::shared_ptr<PendingItem> pendingItem;
  {
    std::lock_guard<std::mutex> lock(seq->mutex);
    if (seq->stopRequested)
      return 0;
    seqrHandle = seq->parentSequencer;
    pendingItem = seq->currentItem;
  }

  if (!pendingItem)
    return 0;

  auto *seqr = registry.getSequencer(seqrHandle);
  if (!seqr)
    return 0;

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Sequence '%s' calling finish_item\n",
                seq->name.c_str());
  }

  // Update the item data in case it was modified
  if (item && itemSize > 0 && static_cast<size_t>(itemSize) == pendingItem->data.size()) {
    std::memcpy(pendingItem->data.data(), item, itemSize);
  }

  // Mark item as ready for the driver
  {
    std::lock_guard<std::mutex> seqLock(seq->mutex);
    pendingItem->itemReady = true;
    seq->waitingForItemDone = true;
    seq->itemDoneSignaled = false;
  }

  // Notify driver that item is ready (driver waits on seq->cv)
  seq->cv.notify_all();

  // Wait for item_done
  {
    std::unique_lock<std::mutex> lock(seq->mutex);
    seq->cv.wait(lock, [&seq]() {
      return seq->itemDoneSignaled || seq->stopRequested;
    });

    seq->waitingForItemDone = false;

    if (seq->stopRequested) {
      return 0;
    }

    // Copy response if available
    if (!pendingItem->response.empty() && item &&
        pendingItem->response.size() <= static_cast<size_t>(itemSize)) {
      std::memcpy(item, pendingItem->response.data(), pendingItem->response.size());
    }
  }

  registry.totalItemsTransferred++;

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Sequence '%s' finish_item complete\n",
                seq->name.c_str());
  }

  return 1;
}

extern "C" int32_t __moore_sequencer_get_next_item(MooreSequencerHandle sequencer,
                                                    void *item, int64_t itemSize) {
  auto &registry = getSeqRegistry();
  auto *seqr = registry.getSequencer(sequencer);
  if (!seqr || !item || itemSize <= 0)
    return 0;

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Driver calling get_next_item on '%s'\n",
                seqr->name.c_str());
  }

  std::unique_lock<std::mutex> seqrLock(seqr->mutex);

  // If there's already an active item (from peek), return it
  if (seqr->hasActiveItem && seqr->activeItem) {
    auto &data = seqr->activeItem->data;
    if (data.size() <= static_cast<size_t>(itemSize)) {
      std::memcpy(item, data.data(), data.size());
    } else {
      std::memcpy(item, data.data(), itemSize);
    }
    return 1;
  }

  // Wait for a sequence to be waiting
  seqr->cv.wait(seqrLock, [&seqr]() {
    return !seqr->waitingSequences.empty() || !seqr->running;
  });

  if (!seqr->running || seqr->waitingSequences.empty())
    return 0;

  // Arbitrate to select which sequence gets access
  MooreSequenceHandle selectedSeq = arbitrateSequence(seqr);
  if (selectedSeq == MOORE_SEQUENCE_INVALID_HANDLE)
    return 0;

  // Remove from waiting list
  auto it = std::find(seqr->waitingSequences.begin(),
                      seqr->waitingSequences.end(), selectedSeq);
  if (it != seqr->waitingSequences.end()) {
    seqr->waitingSequences.erase(it);
  }

  seqr->activeSequence = selectedSeq;
  seqrLock.unlock();

  // Get the sequence and signal driver ready
  auto *seq = registry.getSequence(selectedSeq);
  if (!seq)
    return 0;

  {
    std::lock_guard<std::mutex> seqLock(seq->mutex);
    seq->driverReady = true;
  }
  seq->cv.notify_all();

  // Wait for item to be ready (after finish_item is called)
  std::shared_ptr<PendingItem> pendingItem;
  {
    std::unique_lock<std::mutex> seqLock(seq->mutex);
    seq->cv.wait(seqLock, [&seq]() {
      return (seq->currentItem && seq->currentItem->itemReady) ||
             seq->stopRequested;
    });

    if (seq->stopRequested)
      return 0;

    pendingItem = seq->currentItem;
  }

  if (!pendingItem)
    return 0;

  // Copy item to driver's buffer
  if (pendingItem->data.size() <= static_cast<size_t>(itemSize)) {
    std::memcpy(item, pendingItem->data.data(), pendingItem->data.size());
  } else {
    std::memcpy(item, pendingItem->data.data(), itemSize);
  }

  // Store active item for item_done
  {
    std::lock_guard<std::mutex> seqrLock2(seqr->mutex);
    seqr->activeItem = pendingItem;
    seqr->hasActiveItem = true;
  }

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] Driver received item from sequence '%s'\n",
                seq->name.c_str());
  }

  return 1;
}

extern "C" int32_t
__moore_sequencer_try_get_next_item(MooreSequencerHandle sequencer, void *item,
                                     int64_t itemSize) {
  auto &registry = getSeqRegistry();
  auto *seqr = registry.getSequencer(sequencer);
  if (!seqr || !item || itemSize <= 0)
    return 0;

  MooreSequenceHandle selectedSeq;

  // Non-blocking check: return immediately if no sequences waiting
  {
    std::unique_lock<std::mutex> seqrLock(seqr->mutex);

    if (seqr->waitingSequences.empty())
      return 0;

    // Arbitrate to select which sequence gets access (like get_next_item)
    selectedSeq = arbitrateSequence(seqr);
    if (selectedSeq == MOORE_SEQUENCE_INVALID_HANDLE)
      return 0;

    // Remove from waiting list
    auto it = std::find(seqr->waitingSequences.begin(),
                        seqr->waitingSequences.end(), selectedSeq);
    if (it != seqr->waitingSequences.end()) {
      seqr->waitingSequences.erase(it);
    }

    seqr->activeSequence = selectedSeq;
  }

  // Get the sequence and signal driver ready (like get_next_item)
  auto *seq = registry.getSequence(selectedSeq);
  if (!seq)
    return 0;

  {
    std::lock_guard<std::mutex> seqLock(seq->mutex);
    seq->driverReady = true;
  }
  seq->cv.notify_all();

  // Wait for item to be ready (after finish_item is called)
  std::shared_ptr<PendingItem> pendingItem;
  {
    std::unique_lock<std::mutex> seqLock(seq->mutex);
    seq->cv.wait(seqLock, [&seq]() {
      return (seq->currentItem && seq->currentItem->itemReady) ||
             seq->stopRequested;
    });

    if (seq->stopRequested)
      return 0;

    pendingItem = seq->currentItem;
  }

  if (!pendingItem)
    return 0;

  // Copy item to driver's buffer
  if (pendingItem->data.size() <= static_cast<size_t>(itemSize)) {
    std::memcpy(item, pendingItem->data.data(), pendingItem->data.size());
  } else {
    std::memcpy(item, pendingItem->data.data(), itemSize);
  }

  // Store active item for item_done
  {
    std::lock_guard<std::mutex> seqrLock(seqr->mutex);
    seqr->activeItem = pendingItem;
    seqr->hasActiveItem = true;
  }

  return 1;
}

extern "C" void __moore_sequencer_item_done(MooreSequencerHandle sequencer) {
  auto &registry = getSeqRegistry();
  auto *seqr = registry.getSequencer(sequencer);
  if (!seqr)
    return;

  MooreSequenceHandle activeSeq;
  std::shared_ptr<PendingItem> activeItem;

  {
    std::lock_guard<std::mutex> lock(seqr->mutex);
    if (!seqr->hasActiveItem)
      return;

    activeSeq = seqr->activeSequence;
    activeItem = seqr->activeItem;
    seqr->hasActiveItem = false;
    seqr->activeItem = nullptr;
    seqr->activeSequence = MOORE_SEQUENCE_INVALID_HANDLE;
  }

  auto *seq = registry.getSequence(activeSeq);
  if (!seq)
    return;

  {
    std::lock_guard<std::mutex> seqLock(seq->mutex);
    seq->itemDoneSignaled = true;
    activeItem->itemDone = true;
    seq->currentItem = nullptr;
  }
  seq->cv.notify_all();

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] item_done signaled for sequence '%s'\n",
                seq->name.c_str());
  }
}

extern "C" void
__moore_sequencer_item_done_with_response(MooreSequencerHandle sequencer,
                                           void *response, int64_t responseSize) {
  auto &registry = getSeqRegistry();
  auto *seqr = registry.getSequencer(sequencer);
  if (!seqr)
    return;

  MooreSequenceHandle activeSeq;
  std::shared_ptr<PendingItem> activeItem;

  {
    std::lock_guard<std::mutex> lock(seqr->mutex);
    if (!seqr->hasActiveItem)
      return;

    activeSeq = seqr->activeSequence;
    activeItem = seqr->activeItem;
    seqr->hasActiveItem = false;
    seqr->activeItem = nullptr;
    seqr->activeSequence = MOORE_SEQUENCE_INVALID_HANDLE;
  }

  auto *seq = registry.getSequence(activeSeq);
  if (!seq)
    return;

  // Copy response data
  if (response && responseSize > 0) {
    activeItem->response.resize(responseSize);
    std::memcpy(activeItem->response.data(), response, responseSize);
  }

  {
    std::lock_guard<std::mutex> seqLock(seq->mutex);
    seq->itemDoneSignaled = true;
    activeItem->itemDone = true;
    seq->currentItem = nullptr;
  }
  seq->cv.notify_all();

  if (seqTraceEnabled) {
    std::printf("[SEQ TRACE] item_done_with_response signaled for sequence '%s'\n",
                seq->name.c_str());
  }
}

extern "C" int32_t
__moore_sequencer_peek_next_item(MooreSequencerHandle sequencer, void *item,
                                  int64_t itemSize) {
  auto &registry = getSeqRegistry();
  auto *seqr = registry.getSequencer(sequencer);
  if (!seqr || !item || itemSize <= 0)
    return 0;

  // Check active item first (no lock needed for read)
  {
    std::lock_guard<std::mutex> seqrLock(seqr->mutex);
    if (seqr->hasActiveItem && seqr->activeItem) {
      auto &data = seqr->activeItem->data;
      if (data.size() <= static_cast<size_t>(itemSize)) {
        std::memcpy(item, data.data(), data.size());
      } else {
        std::memcpy(item, data.data(), itemSize);
      }
      return 1;
    }

    // No active item and no waiting sequences
    if (seqr->waitingSequences.empty())
      return 0;
  }

  // A sequence is waiting - trigger handshake to peek at item
  // Use same logic as try_get_next_item but store as active so peek can be
  // followed by get
  MooreSequenceHandle selectedSeq;

  {
    std::unique_lock<std::mutex> seqrLock(seqr->mutex);

    if (seqr->waitingSequences.empty())
      return 0;

    // Arbitrate to select which sequence gets access
    selectedSeq = arbitrateSequence(seqr);
    if (selectedSeq == MOORE_SEQUENCE_INVALID_HANDLE)
      return 0;

    // Remove from waiting list
    auto it = std::find(seqr->waitingSequences.begin(),
                        seqr->waitingSequences.end(), selectedSeq);
    if (it != seqr->waitingSequences.end()) {
      seqr->waitingSequences.erase(it);
    }

    seqr->activeSequence = selectedSeq;
  }

  // Signal driver ready to unblock sequence
  auto *seq = registry.getSequence(selectedSeq);
  if (!seq)
    return 0;

  {
    std::lock_guard<std::mutex> seqLock(seq->mutex);
    seq->driverReady = true;
  }
  seq->cv.notify_all();

  // Wait for item to be ready
  std::shared_ptr<PendingItem> pendingItem;
  {
    std::unique_lock<std::mutex> seqLock(seq->mutex);
    seq->cv.wait(seqLock, [&seq]() {
      return (seq->currentItem && seq->currentItem->itemReady) ||
             seq->stopRequested;
    });

    if (seq->stopRequested)
      return 0;

    pendingItem = seq->currentItem;
  }

  if (!pendingItem)
    return 0;

  // Copy item data
  if (pendingItem->data.size() <= static_cast<size_t>(itemSize)) {
    std::memcpy(item, pendingItem->data.data(), pendingItem->data.size());
  } else {
    std::memcpy(item, pendingItem->data.data(), itemSize);
  }

  // Store as active item (so subsequent get returns same item)
  {
    std::lock_guard<std::mutex> seqrLock(seqr->mutex);
    seqr->activeItem = pendingItem;
    seqr->hasActiveItem = true;
  }

  return 1;
}

extern "C" int32_t __moore_sequencer_has_items(MooreSequencerHandle sequencer) {
  auto *seqr = getSeqRegistry().getSequencer(sequencer);
  if (!seqr)
    return 0;

  std::lock_guard<std::mutex> lock(seqr->mutex);
  return (seqr->hasActiveItem || !seqr->waitingSequences.empty()) ? 1 : 0;
}

//===----------------------------------------------------------------------===//
// Sequence/Sequencer Debugging
//===----------------------------------------------------------------------===//

extern "C" void __moore_seq_set_trace_enabled(int32_t enable) {
  seqTraceEnabled = (enable != 0);
}

extern "C" int32_t __moore_seq_is_trace_enabled(void) {
  return seqTraceEnabled ? 1 : 0;
}

extern "C" void __moore_seq_print_summary(void) {
  auto &registry = getSeqRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);

  std::printf("\n=== UVM Sequence/Sequencer Summary ===\n");

  std::printf("\nSequencers:\n");
  if (registry.sequencers.empty()) {
    std::printf("  No sequencers registered.\n");
  } else {
    for (size_t i = 0; i < registry.sequencers.size(); ++i) {
      auto &seqr = registry.sequencers[i];
      if (seqr) {
        std::lock_guard<std::mutex> seqrLock(seqr->mutex);
        std::printf("  [%zu] '%s': %s, arb=%d, waiting=%zu\n",
                    i, seqr->name.c_str(),
                    seqr->running ? "running" : "stopped",
                    seqr->arbMode,
                    seqr->waitingSequences.size());
      }
    }
  }

  std::printf("\nSequences:\n");
  if (registry.sequences.empty()) {
    std::printf("  No sequences registered.\n");
  } else {
    for (size_t i = 0; i < registry.sequences.size(); ++i) {
      auto &seq = registry.sequences[i];
      if (seq) {
        std::lock_guard<std::mutex> seqLock(seq->mutex);
        const char *stateStr = "unknown";
        switch (seq->state) {
          case MOORE_SEQ_STATE_IDLE: stateStr = "idle"; break;
          case MOORE_SEQ_STATE_RUNNING: stateStr = "running"; break;
          case MOORE_SEQ_STATE_WAITING: stateStr = "waiting"; break;
          case MOORE_SEQ_STATE_FINISHED: stateStr = "finished"; break;
          case MOORE_SEQ_STATE_STOPPED: stateStr = "stopped"; break;
        }
        std::printf("  [%zu] '%s': state=%s, priority=%d\n",
                    i, seq->name.c_str(), stateStr, seq->priority);
      }
    }
  }

  std::printf("\nStatistics:\n");
  std::printf("  Total sequences created: %lld\n",
              (long long)registry.totalSequencesCreated.load());
  std::printf("  Total items transferred: %lld\n",
              (long long)registry.totalItemsTransferred.load());
  std::printf("  Total arbitrations: %lld\n",
              (long long)registry.totalArbitrations.load());

  std::printf("\n======================================\n");
}

extern "C" void __moore_seq_get_statistics(int64_t *totalSequences,
                                            int64_t *totalItems,
                                            int64_t *totalArbitrations) {
  auto &registry = getSeqRegistry();
  if (totalSequences)
    *totalSequences = registry.totalSequencesCreated.load();
  if (totalItems)
    *totalItems = registry.totalItemsTransferred.load();
  if (totalArbitrations)
    *totalArbitrations = registry.totalArbitrations.load();
}

//===----------------------------------------------------------------------===//
// UVM Scoreboard Utilities Implementation
//===----------------------------------------------------------------------===//

namespace {

/// Global statistics for scoreboard operations
static std::atomic<bool> scoreboardTraceEnabled{false};
static std::atomic<int64_t> scoreboardTotalCreated{0};
static std::atomic<int64_t> scoreboardTotalComparisons{0};
static std::atomic<int64_t> scoreboardTotalMatches{0};
static std::atomic<int64_t> scoreboardTotalMismatches{0};

/// Scoreboard structure
struct Scoreboard {
  std::string name;
  int64_t transactionSize;

  // Transaction FIFOs
  std::vector<std::vector<uint8_t>> expectedQueue;
  std::vector<std::vector<uint8_t>> actualQueue;

  // TLM analysis exports for receiving transactions
  MooreTlmFifoHandle expectedFifo;
  MooreTlmFifoHandle actualFifo;

  // Comparison callback
  MooreScoreboardCompareCallback compareCallback;
  void *compareCallbackUserData;

  // Mismatch callback
  MooreScoreboardMismatchCallback mismatchCallback;
  void *mismatchCallbackUserData;

  // Statistics
  std::atomic<int64_t> matchCount{0};
  std::atomic<int64_t> mismatchCount{0};

  // Synchronization
  std::mutex mutex;
  std::condition_variable expectedAvailable;
  std::condition_variable actualAvailable;

  Scoreboard(const std::string &n, int64_t txSize)
      : name(n), transactionSize(txSize),
        expectedFifo(MOORE_TLM_INVALID_HANDLE),
        actualFifo(MOORE_TLM_INVALID_HANDLE),
        compareCallback(nullptr), compareCallbackUserData(nullptr),
        mismatchCallback(nullptr), mismatchCallbackUserData(nullptr) {}
};

/// Global scoreboard registry
struct ScoreboardRegistry {
  std::vector<std::unique_ptr<Scoreboard>> scoreboards;
  std::mutex mutex;

  Scoreboard *getScoreboard(MooreScoreboardHandle handle) {
    std::lock_guard<std::mutex> lock(mutex);
    if (handle < 0 || static_cast<size_t>(handle) >= scoreboards.size())
      return nullptr;
    return scoreboards[handle].get();
  }

  MooreScoreboardHandle addScoreboard(std::unique_ptr<Scoreboard> sb) {
    std::lock_guard<std::mutex> lock(mutex);
    MooreScoreboardHandle handle =
        static_cast<MooreScoreboardHandle>(scoreboards.size());
    scoreboards.push_back(std::move(sb));
    return handle;
  }
};

ScoreboardRegistry &getScoreboardRegistry() {
  static ScoreboardRegistry registry;
  return registry;
}

void scoreboardTrace(const char *fmt, ...) {
  if (!scoreboardTraceEnabled)
    return;
  va_list args;
  va_start(args, fmt);
  std::printf("[SCOREBOARD] ");
  std::vprintf(fmt, args);
  std::printf("\n");
  va_end(args);
}

/// Default byte-by-byte comparison function
int32_t defaultScoreboardCompare(const void *expected, const void *actual,
                                  int64_t transactionSize, void *userData) {
  (void)userData;
  return std::memcmp(expected, actual, transactionSize) == 0 ? 1 : 0;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Scoreboard Creation and Configuration
//===----------------------------------------------------------------------===//

extern "C" MooreScoreboardHandle __moore_scoreboard_create(const char *name,
                                                           int64_t nameLen,
                                                           int64_t transactionSize) {
  std::string sbName(name, nameLen);
  auto scoreboard = std::make_unique<Scoreboard>(sbName, transactionSize);

  // Create TLM FIFOs for expected and actual transactions
  std::string expectedFifoName = sbName + ".expected_fifo";
  std::string actualFifoName = sbName + ".actual_fifo";

  scoreboard->expectedFifo = __moore_tlm_fifo_create(
      expectedFifoName.c_str(), expectedFifoName.size(), 0, 0, transactionSize);
  scoreboard->actualFifo = __moore_tlm_fifo_create(
      actualFifoName.c_str(), actualFifoName.size(), 0, 0, transactionSize);

  scoreboardTotalCreated++;

  scoreboardTrace("Created scoreboard '%s' (transactionSize=%lld)",
                  sbName.c_str(), (long long)transactionSize);

  return getScoreboardRegistry().addScoreboard(std::move(scoreboard));
}

extern "C" void __moore_scoreboard_destroy(MooreScoreboardHandle scoreboard) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (sb) {
    scoreboardTrace("Destroyed scoreboard '%s'", sb->name.c_str());
    // Destroy the underlying FIFOs
    if (sb->expectedFifo != MOORE_TLM_INVALID_HANDLE)
      __moore_tlm_fifo_destroy(sb->expectedFifo);
    if (sb->actualFifo != MOORE_TLM_INVALID_HANDLE)
      __moore_tlm_fifo_destroy(sb->actualFifo);
  }
}

extern "C" void __moore_scoreboard_set_compare_callback(
    MooreScoreboardHandle scoreboard,
    MooreScoreboardCompareCallback callback,
    void *userData) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb)
    return;

  std::lock_guard<std::mutex> lock(sb->mutex);
  sb->compareCallback = callback;
  sb->compareCallbackUserData = userData;

  scoreboardTrace("Set compare callback for scoreboard '%s'", sb->name.c_str());
}

extern "C" void __moore_scoreboard_set_mismatch_callback(
    MooreScoreboardHandle scoreboard,
    MooreScoreboardMismatchCallback callback,
    void *userData) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb)
    return;

  std::lock_guard<std::mutex> lock(sb->mutex);
  sb->mismatchCallback = callback;
  sb->mismatchCallbackUserData = userData;

  scoreboardTrace("Set mismatch callback for scoreboard '%s'", sb->name.c_str());
}

extern "C" MooreString __moore_scoreboard_get_name(MooreScoreboardHandle scoreboard) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb) {
    MooreString empty = {nullptr, 0};
    return empty;
  }

  MooreString result;
  result.len = sb->name.size();
  result.data = static_cast<char *>(std::malloc(result.len));
  if (result.data) {
    std::memcpy(result.data, sb->name.c_str(), result.len);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Scoreboard Transaction Operations
//===----------------------------------------------------------------------===//

extern "C" void __moore_scoreboard_add_expected(MooreScoreboardHandle scoreboard,
                                                 void *transaction,
                                                 int64_t transactionSize) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb)
    return;

  std::lock_guard<std::mutex> lock(sb->mutex);

  // Copy transaction data
  std::vector<uint8_t> txData(static_cast<uint8_t *>(transaction),
                              static_cast<uint8_t *>(transaction) + transactionSize);
  sb->expectedQueue.push_back(std::move(txData));

  // Notify any waiting comparisons
  sb->expectedAvailable.notify_one();

  scoreboardTrace("Added expected transaction to '%s' (queue size=%zu)",
                  sb->name.c_str(), sb->expectedQueue.size());
}

extern "C" void __moore_scoreboard_add_actual(MooreScoreboardHandle scoreboard,
                                               void *transaction,
                                               int64_t transactionSize) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb)
    return;

  std::lock_guard<std::mutex> lock(sb->mutex);

  // Copy transaction data
  std::vector<uint8_t> txData(static_cast<uint8_t *>(transaction),
                              static_cast<uint8_t *>(transaction) + transactionSize);
  sb->actualQueue.push_back(std::move(txData));

  // Notify any waiting comparisons
  sb->actualAvailable.notify_one();

  scoreboardTrace("Added actual transaction to '%s' (queue size=%zu)",
                  sb->name.c_str(), sb->actualQueue.size());
}

extern "C" MooreScoreboardCompareResult
__moore_scoreboard_compare(MooreScoreboardHandle scoreboard) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb)
    return MOORE_SCOREBOARD_TIMEOUT;

  std::unique_lock<std::mutex> lock(sb->mutex);

  // Wait for expected transaction
  while (sb->expectedQueue.empty()) {
    sb->expectedAvailable.wait(lock);
  }

  // Wait for actual transaction
  while (sb->actualQueue.empty()) {
    sb->actualAvailable.wait(lock);
  }

  // Get the transactions
  std::vector<uint8_t> expected = std::move(sb->expectedQueue.front());
  sb->expectedQueue.erase(sb->expectedQueue.begin());

  std::vector<uint8_t> actual = std::move(sb->actualQueue.front());
  sb->actualQueue.erase(sb->actualQueue.begin());

  // Get the comparison function
  MooreScoreboardCompareCallback compareFn = sb->compareCallback;
  void *compareUserData = sb->compareCallbackUserData;
  MooreScoreboardMismatchCallback mismatchFn = sb->mismatchCallback;
  void *mismatchUserData = sb->mismatchCallbackUserData;

  lock.unlock();

  // Perform comparison
  int32_t match;
  if (compareFn) {
    match = compareFn(expected.data(), actual.data(),
                      sb->transactionSize, compareUserData);
  } else {
    match = defaultScoreboardCompare(expected.data(), actual.data(),
                                      sb->transactionSize, nullptr);
  }

  // Update statistics
  scoreboardTotalComparisons++;
  if (match) {
    sb->matchCount++;
    scoreboardTotalMatches++;
    scoreboardTrace("Comparison MATCH in '%s'", sb->name.c_str());
    return MOORE_SCOREBOARD_MATCH;
  } else {
    sb->mismatchCount++;
    scoreboardTotalMismatches++;
    scoreboardTrace("Comparison MISMATCH in '%s'", sb->name.c_str());

    // Call mismatch callback if registered
    if (mismatchFn) {
      mismatchFn(expected.data(), actual.data(),
                 sb->transactionSize, mismatchUserData);
    }

    return MOORE_SCOREBOARD_MISMATCH;
  }
}

extern "C" MooreScoreboardCompareResult
__moore_scoreboard_try_compare(MooreScoreboardHandle scoreboard) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb)
    return MOORE_SCOREBOARD_TIMEOUT;

  std::lock_guard<std::mutex> lock(sb->mutex);

  // Check if both transactions are available
  if (sb->expectedQueue.empty() || sb->actualQueue.empty()) {
    return MOORE_SCOREBOARD_TIMEOUT;
  }

  // Get the transactions
  std::vector<uint8_t> expected = std::move(sb->expectedQueue.front());
  sb->expectedQueue.erase(sb->expectedQueue.begin());

  std::vector<uint8_t> actual = std::move(sb->actualQueue.front());
  sb->actualQueue.erase(sb->actualQueue.begin());

  // Perform comparison
  int32_t match;
  if (sb->compareCallback) {
    match = sb->compareCallback(expected.data(), actual.data(),
                                 sb->transactionSize, sb->compareCallbackUserData);
  } else {
    match = defaultScoreboardCompare(expected.data(), actual.data(),
                                      sb->transactionSize, nullptr);
  }

  // Update statistics
  scoreboardTotalComparisons++;
  if (match) {
    sb->matchCount++;
    scoreboardTotalMatches++;
    scoreboardTrace("Try comparison MATCH in '%s'", sb->name.c_str());
    return MOORE_SCOREBOARD_MATCH;
  } else {
    sb->mismatchCount++;
    scoreboardTotalMismatches++;
    scoreboardTrace("Try comparison MISMATCH in '%s'", sb->name.c_str());

    // Call mismatch callback if registered
    if (sb->mismatchCallback) {
      sb->mismatchCallback(expected.data(), actual.data(),
                           sb->transactionSize, sb->mismatchCallbackUserData);
    }

    return MOORE_SCOREBOARD_MISMATCH;
  }
}

extern "C" int64_t __moore_scoreboard_compare_all(MooreScoreboardHandle scoreboard) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb)
    return 0;

  int64_t comparisons = 0;

  while (true) {
    MooreScoreboardCompareResult result = __moore_scoreboard_try_compare(scoreboard);
    if (result == MOORE_SCOREBOARD_TIMEOUT) {
      break;
    }
    comparisons++;
  }

  scoreboardTrace("Compared all: %lld comparisons in '%s'",
                  (long long)comparisons, sb->name.c_str());

  return comparisons;
}

//===----------------------------------------------------------------------===//
// Scoreboard TLM Integration
//===----------------------------------------------------------------------===//

extern "C" MooreTlmPortHandle
__moore_scoreboard_get_expected_export(MooreScoreboardHandle scoreboard) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb || sb->expectedFifo == MOORE_TLM_INVALID_HANDLE)
    return MOORE_TLM_INVALID_HANDLE;

  return __moore_tlm_fifo_get_analysis_export(sb->expectedFifo);
}

extern "C" MooreTlmPortHandle
__moore_scoreboard_get_actual_export(MooreScoreboardHandle scoreboard) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb || sb->actualFifo == MOORE_TLM_INVALID_HANDLE)
    return MOORE_TLM_INVALID_HANDLE;

  return __moore_tlm_fifo_get_analysis_export(sb->actualFifo);
}

//===----------------------------------------------------------------------===//
// Scoreboard Statistics and Reporting
//===----------------------------------------------------------------------===//

extern "C" int64_t __moore_scoreboard_get_match_count(MooreScoreboardHandle scoreboard) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb)
    return 0;
  return sb->matchCount.load();
}

extern "C" int64_t __moore_scoreboard_get_mismatch_count(MooreScoreboardHandle scoreboard) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb)
    return 0;
  return sb->mismatchCount.load();
}

extern "C" int64_t __moore_scoreboard_get_pending_expected(MooreScoreboardHandle scoreboard) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb)
    return 0;

  std::lock_guard<std::mutex> lock(sb->mutex);
  return static_cast<int64_t>(sb->expectedQueue.size());
}

extern "C" int64_t __moore_scoreboard_get_pending_actual(MooreScoreboardHandle scoreboard) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb)
    return 0;

  std::lock_guard<std::mutex> lock(sb->mutex);
  return static_cast<int64_t>(sb->actualQueue.size());
}

extern "C" int32_t __moore_scoreboard_is_empty(MooreScoreboardHandle scoreboard) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb)
    return 1;

  std::lock_guard<std::mutex> lock(sb->mutex);
  return (sb->expectedQueue.empty() && sb->actualQueue.empty()) ? 1 : 0;
}

extern "C" void __moore_scoreboard_report(MooreScoreboardHandle scoreboard) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb) {
    std::printf("[SCOREBOARD] Error: Invalid scoreboard handle\n");
    return;
  }

  std::lock_guard<std::mutex> lock(sb->mutex);

  int64_t matches = sb->matchCount.load();
  int64_t mismatches = sb->mismatchCount.load();
  int64_t pendingExpected = static_cast<int64_t>(sb->expectedQueue.size());
  int64_t pendingActual = static_cast<int64_t>(sb->actualQueue.size());
  bool passed = (mismatches == 0 && pendingExpected == 0 && pendingActual == 0);

  std::printf("\n========================================\n");
  std::printf("Scoreboard Report: %s\n", sb->name.c_str());
  std::printf("========================================\n");
  std::printf("  Matches:          %lld\n", (long long)matches);
  std::printf("  Mismatches:       %lld\n", (long long)mismatches);
  std::printf("  Pending Expected: %lld\n", (long long)pendingExpected);
  std::printf("  Pending Actual:   %lld\n", (long long)pendingActual);
  std::printf("  Status:           %s\n", passed ? "PASSED" : "FAILED");
  std::printf("========================================\n\n");
}

extern "C" int32_t __moore_scoreboard_passed(MooreScoreboardHandle scoreboard) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb)
    return 0;

  std::lock_guard<std::mutex> lock(sb->mutex);

  // A scoreboard passes if:
  // - No mismatches
  // - No pending expected transactions
  // - No pending actual transactions
  return (sb->mismatchCount.load() == 0 &&
          sb->expectedQueue.empty() &&
          sb->actualQueue.empty()) ? 1 : 0;
}

extern "C" void __moore_scoreboard_reset(MooreScoreboardHandle scoreboard) {
  auto *sb = getScoreboardRegistry().getScoreboard(scoreboard);
  if (!sb)
    return;

  std::lock_guard<std::mutex> lock(sb->mutex);

  sb->expectedQueue.clear();
  sb->actualQueue.clear();
  sb->matchCount = 0;
  sb->mismatchCount = 0;

  // Also flush the TLM FIFOs
  if (sb->expectedFifo != MOORE_TLM_INVALID_HANDLE)
    __moore_tlm_fifo_flush(sb->expectedFifo);
  if (sb->actualFifo != MOORE_TLM_INVALID_HANDLE)
    __moore_tlm_fifo_flush(sb->actualFifo);

  scoreboardTrace("Reset scoreboard '%s'", sb->name.c_str());
}

//===----------------------------------------------------------------------===//
// Scoreboard Debugging/Tracing
//===----------------------------------------------------------------------===//

extern "C" void __moore_scoreboard_set_trace_enabled(int32_t enable) {
  scoreboardTraceEnabled = (enable != 0);
}

extern "C" int32_t __moore_scoreboard_is_trace_enabled(void) {
  return scoreboardTraceEnabled ? 1 : 0;
}

extern "C" void __moore_scoreboard_print_summary(void) {
  auto &registry = getScoreboardRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);

  std::printf("\n=== UVM Scoreboard Summary ===\n");

  if (registry.scoreboards.empty()) {
    std::printf("  No scoreboards registered.\n");
  } else {
    for (size_t i = 0; i < registry.scoreboards.size(); ++i) {
      auto &sb = registry.scoreboards[i];
      if (sb) {
        std::lock_guard<std::mutex> sbLock(sb->mutex);
        int64_t matches = sb->matchCount.load();
        int64_t mismatches = sb->mismatchCount.load();
        int64_t pendingExp = static_cast<int64_t>(sb->expectedQueue.size());
        int64_t pendingAct = static_cast<int64_t>(sb->actualQueue.size());
        bool passed = (mismatches == 0 && pendingExp == 0 && pendingAct == 0);

        std::printf("  [%zu] '%s': matches=%lld, mismatches=%lld, "
                    "pending_exp=%lld, pending_act=%lld, %s\n",
                    i, sb->name.c_str(),
                    (long long)matches, (long long)mismatches,
                    (long long)pendingExp, (long long)pendingAct,
                    passed ? "PASSED" : "FAILED");
      }
    }
  }

  std::printf("\nGlobal Statistics:\n");
  std::printf("  Total scoreboards created: %lld\n",
              (long long)scoreboardTotalCreated.load());
  std::printf("  Total comparisons: %lld\n",
              (long long)scoreboardTotalComparisons.load());
  std::printf("  Total matches: %lld\n",
              (long long)scoreboardTotalMatches.load());
  std::printf("  Total mismatches: %lld\n",
              (long long)scoreboardTotalMismatches.load());

  std::printf("\n==============================\n");
}

extern "C" void __moore_scoreboard_get_statistics(int64_t *totalScoreboards,
                                                   int64_t *totalComparisons,
                                                   int64_t *totalMatches,
                                                   int64_t *totalMismatches) {
  if (totalScoreboards)
    *totalScoreboards = scoreboardTotalCreated.load();
  if (totalComparisons)
    *totalComparisons = scoreboardTotalComparisons.load();
  if (totalMatches)
    *totalMatches = scoreboardTotalMatches.load();
  if (totalMismatches)
    *totalMismatches = scoreboardTotalMismatches.load();
}

//===----------------------------------------------------------------------===//
// UVM Register Abstraction Layer (RAL) Infrastructure
//===----------------------------------------------------------------------===//

namespace {

// Global RAL tracing flag
static bool ralTraceEnabled = false;

// Global RAL statistics
static std::atomic<int64_t> ralTotalRegs{0};
static std::atomic<int64_t> ralTotalReads{0};
static std::atomic<int64_t> ralTotalWrites{0};

/// Register field structure
struct RegField {
  std::string name;
  int32_t numBits;
  int32_t lsbPos;
  MooreRegAccessPolicy access;
  uint64_t resetValue;

  RegField(const std::string &n, int32_t bits, int32_t lsb,
           MooreRegAccessPolicy acc, uint64_t reset)
      : name(n), numBits(bits), lsbPos(lsb), access(acc), resetValue(reset) {}

  uint64_t getMask() const {
    if (numBits >= 64)
      return ~0ULL;
    return ((1ULL << numBits) - 1) << lsbPos;
  }

  uint64_t extractValue(uint64_t regValue) const {
    return (regValue >> lsbPos) & ((1ULL << numBits) - 1);
  }

  uint64_t insertValue(uint64_t regValue, uint64_t fieldValue) const {
    uint64_t mask = getMask();
    uint64_t shiftedValue = (fieldValue << lsbPos) & mask;
    return (regValue & ~mask) | shiftedValue;
  }
};

/// Register structure
struct Register {
  std::string name;
  int32_t numBits;

  // Values
  uint64_t mirrorValue;      // Current predicted value
  uint64_t desiredValue;     // Value to write on update
  uint64_t hardResetValue;   // Value after hard reset
  uint64_t softResetValue;   // Value after soft reset

  // Fields
  std::vector<std::unique_ptr<RegField>> fields;

  // Address mapping (map handle -> offset)
  std::map<MooreRegMapHandle, uint64_t> mapOffsets;

  // Access callback
  MooreRegAccessCallback accessCallback;
  void *accessCallbackUserData;

  // Parent block
  MooreRegBlockHandle parentBlock;

  // Write-once tracking
  bool hasBeenWritten;

  Register(const std::string &n, int32_t bits)
      : name(n), numBits(bits), mirrorValue(0), desiredValue(0),
        hardResetValue(0), softResetValue(0), accessCallback(nullptr),
        accessCallbackUserData(nullptr), parentBlock(MOORE_REG_INVALID_HANDLE),
        hasBeenWritten(false) {}

  uint64_t getMask() const {
    if (numBits >= 64)
      return ~0ULL;
    return (1ULL << numBits) - 1;
  }
};

/// Register map entry
struct RegMapEntry {
  MooreRegHandle reg;
  uint64_t offset;
  std::string rights;

  RegMapEntry(MooreRegHandle r, uint64_t off, const std::string &r_str)
      : reg(r), offset(off), rights(r_str) {}
};

/// Register map structure
struct RegMap {
  std::string name;
  uint64_t baseAddr;
  int32_t nBytes;      // Bus width in bytes
  int32_t endian;      // 0=little, 1=big

  // Register entries (offset -> entry)
  std::vector<RegMapEntry> entries;

  // Sub-maps (offset -> child map handle)
  std::vector<std::pair<uint64_t, MooreRegMapHandle>> submaps;

  // Sequencer/adapter for frontdoor access
  int64_t sequencer;
  int64_t adapter;

  // Parent block
  MooreRegBlockHandle parentBlock;

  RegMap(const std::string &n, uint64_t base, int32_t bytes, int32_t end)
      : name(n), baseAddr(base), nBytes(bytes), endian(end),
        sequencer(0), adapter(0), parentBlock(MOORE_REG_INVALID_HANDLE) {}
};

/// Register block entry (register with offset)
struct RegBlockRegEntry {
  MooreRegHandle reg;
  uint64_t offset;

  RegBlockRegEntry(MooreRegHandle r, uint64_t off) : reg(r), offset(off) {}
};

/// Register block sub-block entry
struct RegBlockSubBlockEntry {
  MooreRegBlockHandle block;
  uint64_t offset;

  RegBlockSubBlockEntry(MooreRegBlockHandle b, uint64_t off)
      : block(b), offset(off) {}
};

/// Register block structure
struct RegBlock {
  std::string name;

  // Registers in this block
  std::vector<RegBlockRegEntry> registers;

  // Sub-blocks
  std::vector<RegBlockSubBlockEntry> subBlocks;

  // Maps owned by this block
  std::vector<MooreRegMapHandle> maps;

  // Default map
  MooreRegMapHandle defaultMap;

  // Lock state
  bool locked;

  RegBlock(const std::string &n)
      : name(n), defaultMap(MOORE_REG_INVALID_HANDLE), locked(false) {}
};

/// Global register registry
struct RegRegistry {
  std::vector<std::unique_ptr<Register>> registers;
  std::mutex regMutex;

  Register *getRegister(MooreRegHandle handle) {
    std::lock_guard<std::mutex> lock(regMutex);
    if (handle < 0 || static_cast<size_t>(handle) >= registers.size())
      return nullptr;
    return registers[handle].get();
  }

  MooreRegHandle addRegister(std::unique_ptr<Register> reg) {
    std::lock_guard<std::mutex> lock(regMutex);
    MooreRegHandle handle = static_cast<MooreRegHandle>(registers.size());
    registers.push_back(std::move(reg));
    return handle;
  }
};

/// Global register block registry
struct RegBlockRegistry {
  std::vector<std::unique_ptr<RegBlock>> blocks;
  std::mutex mutex;

  RegBlock *getBlock(MooreRegBlockHandle handle) {
    std::lock_guard<std::mutex> lock(mutex);
    if (handle < 0 || static_cast<size_t>(handle) >= blocks.size())
      return nullptr;
    return blocks[handle].get();
  }

  MooreRegBlockHandle addBlock(std::unique_ptr<RegBlock> block) {
    std::lock_guard<std::mutex> lock(mutex);
    MooreRegBlockHandle handle = static_cast<MooreRegBlockHandle>(blocks.size());
    blocks.push_back(std::move(block));
    return handle;
  }
};

/// Global register map registry
struct RegMapRegistry {
  std::vector<std::unique_ptr<RegMap>> maps;
  std::mutex mutex;

  RegMap *getMap(MooreRegMapHandle handle) {
    std::lock_guard<std::mutex> lock(mutex);
    if (handle < 0 || static_cast<size_t>(handle) >= maps.size())
      return nullptr;
    return maps[handle].get();
  }

  MooreRegMapHandle addMap(std::unique_ptr<RegMap> map) {
    std::lock_guard<std::mutex> lock(mutex);
    MooreRegMapHandle handle = static_cast<MooreRegMapHandle>(maps.size());
    maps.push_back(std::move(map));
    return handle;
  }
};

RegRegistry &getRegRegistry() {
  static RegRegistry registry;
  return registry;
}

RegBlockRegistry &getRegBlockRegistry() {
  static RegBlockRegistry registry;
  return registry;
}

RegMapRegistry &getRegMapRegistry() {
  static RegMapRegistry registry;
  return registry;
}

void ralTrace(const char *fmt, ...) {
  if (!ralTraceEnabled)
    return;
  va_list args;
  va_start(args, fmt);
  std::printf("[RAL] ");
  std::vprintf(fmt, args);
  std::printf("\n");
  va_end(args);
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Register Operations
//===----------------------------------------------------------------------===//

extern "C" MooreRegHandle __moore_reg_create(const char *name, int64_t nameLen,
                                             int32_t numBits) {
  if (!name || nameLen <= 0 || numBits <= 0 || numBits > 64)
    return MOORE_REG_INVALID_HANDLE;

  std::string regName(name, nameLen);
  auto reg = std::make_unique<Register>(regName, numBits);

  MooreRegHandle handle = getRegRegistry().addRegister(std::move(reg));
  ralTotalRegs.fetch_add(1);

  ralTrace("Created register '%s' with %d bits (handle=%lld)",
           regName.c_str(), numBits, (long long)handle);

  return handle;
}

extern "C" void __moore_reg_destroy(MooreRegHandle reg) {
  auto &registry = getRegRegistry();
  std::lock_guard<std::mutex> lock(registry.regMutex);

  if (reg >= 0 && static_cast<size_t>(reg) < registry.registers.size()) {
    ralTrace("Destroyed register handle=%lld", (long long)reg);
    registry.registers[reg].reset();
  }
}

extern "C" MooreString __moore_reg_get_name(MooreRegHandle reg) {
  Register *r = getRegRegistry().getRegister(reg);
  if (!r) {
    return {nullptr, 0};
  }

  char *data = static_cast<char *>(std::malloc(r->name.size()));
  if (!data)
    return {nullptr, 0};

  std::memcpy(data, r->name.data(), r->name.size());
  return {data, static_cast<int64_t>(r->name.size())};
}

extern "C" int32_t __moore_reg_get_n_bits(MooreRegHandle reg) {
  Register *r = getRegRegistry().getRegister(reg);
  return r ? r->numBits : 0;
}

extern "C" uint64_t __moore_reg_get_address(MooreRegHandle reg,
                                            MooreRegMapHandle map) {
  Register *r = getRegRegistry().getRegister(reg);
  if (!r)
    return 0;

  // If specific map requested, get offset from that map
  if (map != MOORE_REG_INVALID_HANDLE) {
    auto it = r->mapOffsets.find(map);
    if (it != r->mapOffsets.end()) {
      RegMap *m = getRegMapRegistry().getMap(map);
      if (m)
        return m->baseAddr + it->second;
    }
  }

  // Otherwise use first available map
  if (!r->mapOffsets.empty()) {
    auto it = r->mapOffsets.begin();
    RegMap *m = getRegMapRegistry().getMap(it->first);
    if (m)
      return m->baseAddr + it->second;
  }

  return 0;
}

extern "C" uint64_t __moore_reg_read(MooreRegHandle reg, MooreRegMapHandle map,
                                     MooreRegPathKind path,
                                     MooreRegStatus *status, int64_t parent) {
  (void)map;
  (void)parent;

  Register *r = getRegRegistry().getRegister(reg);
  if (!r) {
    if (status)
      *status = UVM_REG_STATUS_NOT_OK;
    return 0;
  }

  uint64_t value = r->mirrorValue;

  // For backdoor access, we could read from HDL path if configured
  // For now, just return the mirror value
  if (path == UVM_BACKDOOR) {
    // TODO: Integrate with HDL access functions if HDL path is set
    value = r->mirrorValue;
  }

  // Call access callback if registered
  if (r->accessCallback) {
    r->accessCallback(reg, value, 0, r->accessCallbackUserData);
  }

  ralTotalReads.fetch_add(1);

  ralTrace("Read register '%s': value=0x%llx (path=%d)",
           r->name.c_str(), (unsigned long long)value, (int)path);

  if (status)
    *status = UVM_REG_STATUS_OK;

  return value;
}

extern "C" void __moore_reg_write(MooreRegHandle reg, MooreRegMapHandle map,
                                  uint64_t value, MooreRegPathKind path,
                                  MooreRegStatus *status, int64_t parent) {
  (void)map;
  (void)parent;

  Register *r = getRegRegistry().getRegister(reg);
  if (!r) {
    if (status)
      *status = UVM_REG_STATUS_NOT_OK;
    return;
  }

  // Mask value to register width
  value &= r->getMask();

  // Update mirror value
  r->mirrorValue = value;
  r->hasBeenWritten = true;

  // For backdoor access, we could write to HDL path if configured
  if (path == UVM_BACKDOOR) {
    // TODO: Integrate with HDL access functions if HDL path is set
  }

  // Call access callback if registered
  if (r->accessCallback) {
    r->accessCallback(reg, value, 1, r->accessCallbackUserData);
  }

  ralTotalWrites.fetch_add(1);

  ralTrace("Write register '%s': value=0x%llx (path=%d)",
           r->name.c_str(), (unsigned long long)value, (int)path);

  if (status)
    *status = UVM_REG_STATUS_OK;
}

extern "C" uint64_t __moore_reg_get_value(MooreRegHandle reg) {
  Register *r = getRegRegistry().getRegister(reg);
  return r ? r->mirrorValue : 0;
}

extern "C" void __moore_reg_set_value(MooreRegHandle reg, uint64_t value) {
  Register *r = getRegRegistry().getRegister(reg);
  if (r) {
    r->mirrorValue = value & r->getMask();
    ralTrace("Set register '%s' mirror value to 0x%llx",
             r->name.c_str(), (unsigned long long)r->mirrorValue);
  }
}

extern "C" uint64_t __moore_reg_get_desired(MooreRegHandle reg) {
  Register *r = getRegRegistry().getRegister(reg);
  return r ? r->desiredValue : 0;
}

extern "C" void __moore_reg_set_desired(MooreRegHandle reg, uint64_t value) {
  Register *r = getRegRegistry().getRegister(reg);
  if (r) {
    r->desiredValue = value & r->getMask();
    ralTrace("Set register '%s' desired value to 0x%llx",
             r->name.c_str(), (unsigned long long)r->desiredValue);
  }
}

extern "C" void __moore_reg_update(MooreRegHandle reg, MooreRegMapHandle map,
                                   MooreRegPathKind path,
                                   MooreRegStatus *status) {
  Register *r = getRegRegistry().getRegister(reg);
  if (!r) {
    if (status)
      *status = UVM_REG_STATUS_NOT_OK;
    return;
  }

  // Write desired value to register
  __moore_reg_write(reg, map, r->desiredValue, path, status, 0);
}

extern "C" void __moore_reg_mirror(MooreRegHandle reg, MooreRegMapHandle map,
                                   MooreRegPathKind path,
                                   MooreRegStatus *status) {
  Register *r = getRegRegistry().getRegister(reg);
  if (!r) {
    if (status)
      *status = UVM_REG_STATUS_NOT_OK;
    return;
  }

  // Read actual value and update mirror
  uint64_t value = __moore_reg_read(reg, map, path, status, 0);
  r->mirrorValue = value;
  r->desiredValue = value;

  ralTrace("Mirror register '%s': value=0x%llx",
           r->name.c_str(), (unsigned long long)value);
}

extern "C" bool __moore_reg_predict(MooreRegHandle reg, uint64_t value,
                                    bool isWrite) {
  Register *r = getRegRegistry().getRegister(reg);
  if (!r)
    return false;

  value &= r->getMask();

  if (isWrite) {
    // Predict effect of write
    r->mirrorValue = value;
  }
  // For read, mirror doesn't change (unless special access policy)

  ralTrace("Predict register '%s': value=0x%llx (isWrite=%d)",
           r->name.c_str(), (unsigned long long)value, isWrite);

  return true;
}

extern "C" void __moore_reg_reset(MooreRegHandle reg, const char *kind) {
  Register *r = getRegRegistry().getRegister(reg);
  if (!r)
    return;

  bool isHard = (kind && std::strcmp(kind, "HARD") == 0);

  r->mirrorValue = isHard ? r->hardResetValue : r->softResetValue;
  r->desiredValue = r->mirrorValue;
  r->hasBeenWritten = false;

  ralTrace("Reset register '%s' (%s): value=0x%llx",
           r->name.c_str(), kind ? kind : "HARD",
           (unsigned long long)r->mirrorValue);
}

extern "C" void __moore_reg_set_reset(MooreRegHandle reg, uint64_t value,
                                      const char *kind) {
  Register *r = getRegRegistry().getRegister(reg);
  if (!r)
    return;

  value &= r->getMask();

  bool isHard = (kind && std::strcmp(kind, "HARD") == 0);
  if (isHard) {
    r->hardResetValue = value;
  } else {
    r->softResetValue = value;
  }

  ralTrace("Set reset value for register '%s' (%s): 0x%llx",
           r->name.c_str(), kind ? kind : "HARD", (unsigned long long)value);
}

extern "C" uint64_t __moore_reg_get_reset(MooreRegHandle reg, const char *kind) {
  Register *r = getRegRegistry().getRegister(reg);
  if (!r)
    return 0;

  bool isHard = (kind && std::strcmp(kind, "HARD") == 0);
  return isHard ? r->hardResetValue : r->softResetValue;
}

extern "C" bool __moore_reg_needs_update(MooreRegHandle reg) {
  Register *r = getRegRegistry().getRegister(reg);
  return r ? (r->mirrorValue != r->desiredValue) : false;
}

extern "C" void __moore_reg_set_access_callback(MooreRegHandle reg,
                                                MooreRegAccessCallback callback,
                                                void *userData) {
  Register *r = getRegRegistry().getRegister(reg);
  if (r) {
    r->accessCallback = callback;
    r->accessCallbackUserData = userData;
  }
}

//===----------------------------------------------------------------------===//
// Register Field Operations
//===----------------------------------------------------------------------===//

extern "C" MooreRegFieldHandle __moore_reg_add_field(MooreRegHandle reg,
                                                     const char *name,
                                                     int64_t nameLen,
                                                     int32_t numBits,
                                                     int32_t lsbPos,
                                                     MooreRegAccessPolicy access,
                                                     uint64_t reset) {
  Register *r = getRegRegistry().getRegister(reg);
  if (!r || !name || nameLen <= 0 || numBits <= 0)
    return MOORE_REG_INVALID_HANDLE;

  // Validate field fits within register
  if (lsbPos + numBits > r->numBits)
    return MOORE_REG_INVALID_HANDLE;

  std::string fieldName(name, nameLen);
  auto field = std::make_unique<RegField>(fieldName, numBits, lsbPos,
                                          access, reset);

  MooreRegFieldHandle handle = static_cast<MooreRegFieldHandle>(r->fields.size());
  r->fields.push_back(std::move(field));

  // Update register's reset value with this field's reset value
  r->hardResetValue = r->fields.back()->insertValue(r->hardResetValue, reset);
  r->softResetValue = r->hardResetValue;
  r->mirrorValue = r->hardResetValue;
  r->desiredValue = r->hardResetValue;

  ralTrace("Added field '%s' to register '%s': bits=%d, lsb=%d, reset=0x%llx",
           fieldName.c_str(), r->name.c_str(), numBits, lsbPos,
           (unsigned long long)reset);

  return handle;
}

extern "C" uint64_t __moore_reg_field_get_value(MooreRegHandle reg,
                                                MooreRegFieldHandle field) {
  Register *r = getRegRegistry().getRegister(reg);
  if (!r || field < 0 || static_cast<size_t>(field) >= r->fields.size())
    return 0;

  return r->fields[field]->extractValue(r->mirrorValue);
}

extern "C" void __moore_reg_field_set_value(MooreRegHandle reg,
                                            MooreRegFieldHandle field,
                                            uint64_t value) {
  Register *r = getRegRegistry().getRegister(reg);
  if (!r || field < 0 || static_cast<size_t>(field) >= r->fields.size())
    return;

  r->mirrorValue = r->fields[field]->insertValue(r->mirrorValue, value);

  ralTrace("Set field %lld of register '%s' to 0x%llx (mirror now 0x%llx)",
           (long long)field, r->name.c_str(), (unsigned long long)value,
           (unsigned long long)r->mirrorValue);
}

extern "C" MooreRegFieldHandle __moore_reg_get_field_by_name(MooreRegHandle reg,
                                                             const char *name,
                                                             int64_t nameLen) {
  Register *r = getRegRegistry().getRegister(reg);
  if (!r || !name || nameLen <= 0)
    return MOORE_REG_INVALID_HANDLE;

  std::string fieldName(name, nameLen);
  for (size_t i = 0; i < r->fields.size(); ++i) {
    if (r->fields[i]->name == fieldName)
      return static_cast<MooreRegFieldHandle>(i);
  }

  return MOORE_REG_INVALID_HANDLE;
}

extern "C" int32_t __moore_reg_get_n_fields(MooreRegHandle reg) {
  Register *r = getRegRegistry().getRegister(reg);
  return r ? static_cast<int32_t>(r->fields.size()) : 0;
}

//===----------------------------------------------------------------------===//
// Register Block Operations
//===----------------------------------------------------------------------===//

extern "C" MooreRegBlockHandle __moore_reg_block_create(const char *name,
                                                        int64_t nameLen) {
  if (!name || nameLen <= 0)
    return MOORE_REG_INVALID_HANDLE;

  std::string blockName(name, nameLen);
  auto block = std::make_unique<RegBlock>(blockName);

  MooreRegBlockHandle handle = getRegBlockRegistry().addBlock(std::move(block));

  ralTrace("Created register block '%s' (handle=%lld)",
           blockName.c_str(), (long long)handle);

  return handle;
}

extern "C" void __moore_reg_block_destroy(MooreRegBlockHandle block) {
  auto &registry = getRegBlockRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);

  if (block >= 0 && static_cast<size_t>(block) < registry.blocks.size()) {
    RegBlock *b = registry.blocks[block].get();
    if (b) {
      // Destroy all maps in this block
      for (auto mapHandle : b->maps) {
        __moore_reg_map_destroy(mapHandle);
      }

      // Note: We don't destroy registers here as they may be shared
      // with other blocks. The caller should manage register lifetime.

      ralTrace("Destroyed register block handle=%lld", (long long)block);
    }
    registry.blocks[block].reset();
  }
}

extern "C" MooreString __moore_reg_block_get_name(MooreRegBlockHandle block) {
  RegBlock *b = getRegBlockRegistry().getBlock(block);
  if (!b) {
    return {nullptr, 0};
  }

  char *data = static_cast<char *>(std::malloc(b->name.size()));
  if (!data)
    return {nullptr, 0};

  std::memcpy(data, b->name.data(), b->name.size());
  return {data, static_cast<int64_t>(b->name.size())};
}

extern "C" void __moore_reg_block_add_reg(MooreRegBlockHandle block,
                                          MooreRegHandle reg, uint64_t offset) {
  RegBlock *b = getRegBlockRegistry().getBlock(block);
  Register *r = getRegRegistry().getRegister(reg);

  if (!b || !r)
    return;

  if (b->locked) {
    ralTrace("Cannot add register to locked block '%s'", b->name.c_str());
    return;
  }

  b->registers.emplace_back(reg, offset);
  r->parentBlock = block;

  ralTrace("Added register '%s' to block '%s' at offset 0x%llx",
           r->name.c_str(), b->name.c_str(), (unsigned long long)offset);
}

extern "C" void __moore_reg_block_add_block(MooreRegBlockHandle parent,
                                            MooreRegBlockHandle child,
                                            uint64_t offset) {
  RegBlock *p = getRegBlockRegistry().getBlock(parent);
  RegBlock *c = getRegBlockRegistry().getBlock(child);

  if (!p || !c)
    return;

  if (p->locked) {
    ralTrace("Cannot add sub-block to locked block '%s'", p->name.c_str());
    return;
  }

  p->subBlocks.emplace_back(child, offset);

  ralTrace("Added sub-block '%s' to block '%s' at offset 0x%llx",
           c->name.c_str(), p->name.c_str(), (unsigned long long)offset);
}

extern "C" MooreRegMapHandle __moore_reg_block_get_default_map(
    MooreRegBlockHandle block) {
  RegBlock *b = getRegBlockRegistry().getBlock(block);
  return b ? b->defaultMap : MOORE_REG_INVALID_HANDLE;
}

extern "C" void __moore_reg_block_set_default_map(MooreRegBlockHandle block,
                                                  MooreRegMapHandle map) {
  RegBlock *b = getRegBlockRegistry().getBlock(block);
  if (b) {
    b->defaultMap = map;
    ralTrace("Set default map for block '%s' to handle=%lld",
             b->name.c_str(), (long long)map);
  }
}

extern "C" MooreRegHandle __moore_reg_block_get_reg_by_name(
    MooreRegBlockHandle block, const char *name, int64_t nameLen) {
  RegBlock *b = getRegBlockRegistry().getBlock(block);
  if (!b || !name || nameLen <= 0)
    return MOORE_REG_INVALID_HANDLE;

  std::string regName(name, nameLen);

  // Check for hierarchical name (contains '.')
  size_t dotPos = regName.find('.');
  if (dotPos != std::string::npos) {
    std::string subBlockName = regName.substr(0, dotPos);
    std::string remainingName = regName.substr(dotPos + 1);

    // Find sub-block
    for (const auto &entry : b->subBlocks) {
      RegBlock *subBlock = getRegBlockRegistry().getBlock(entry.block);
      if (subBlock && subBlock->name == subBlockName) {
        return __moore_reg_block_get_reg_by_name(entry.block,
                                                  remainingName.c_str(),
                                                  remainingName.size());
      }
    }
    return MOORE_REG_INVALID_HANDLE;
  }

  // Search in this block's registers
  for (const auto &entry : b->registers) {
    Register *r = getRegRegistry().getRegister(entry.reg);
    if (r && r->name == regName)
      return entry.reg;
  }

  return MOORE_REG_INVALID_HANDLE;
}

extern "C" int32_t __moore_reg_block_get_n_regs(MooreRegBlockHandle block) {
  RegBlock *b = getRegBlockRegistry().getBlock(block);
  return b ? static_cast<int32_t>(b->registers.size()) : 0;
}

extern "C" void __moore_reg_block_lock(MooreRegBlockHandle block) {
  RegBlock *b = getRegBlockRegistry().getBlock(block);
  if (b) {
    b->locked = true;
    ralTrace("Locked register block '%s'", b->name.c_str());
  }
}

extern "C" bool __moore_reg_block_is_locked(MooreRegBlockHandle block) {
  RegBlock *b = getRegBlockRegistry().getBlock(block);
  return b ? b->locked : false;
}

extern "C" void __moore_reg_block_reset(MooreRegBlockHandle block,
                                        const char *kind) {
  RegBlock *b = getRegBlockRegistry().getBlock(block);
  if (!b)
    return;

  // Reset all registers in this block
  for (const auto &entry : b->registers) {
    __moore_reg_reset(entry.reg, kind);
  }

  // Reset all sub-blocks
  for (const auto &entry : b->subBlocks) {
    __moore_reg_block_reset(entry.block, kind);
  }

  ralTrace("Reset register block '%s' (%s)", b->name.c_str(),
           kind ? kind : "HARD");
}

//===----------------------------------------------------------------------===//
// Register Map Operations
//===----------------------------------------------------------------------===//

extern "C" MooreRegMapHandle __moore_reg_map_create(MooreRegBlockHandle block,
                                                    const char *name,
                                                    int64_t nameLen,
                                                    uint64_t baseAddr,
                                                    int32_t nBytes,
                                                    int32_t endian) {
  RegBlock *b = getRegBlockRegistry().getBlock(block);
  if (!b || !name || nameLen <= 0)
    return MOORE_REG_INVALID_HANDLE;

  if (b->locked) {
    ralTrace("Cannot create map in locked block '%s'", b->name.c_str());
    return MOORE_REG_INVALID_HANDLE;
  }

  std::string mapName(name, nameLen);
  auto regMap = std::make_unique<RegMap>(mapName, baseAddr, nBytes, endian);
  regMap->parentBlock = block;

  MooreRegMapHandle handle = getRegMapRegistry().addMap(std::move(regMap));

  // Add map to block's list and set as default if first map
  b->maps.push_back(handle);
  if (b->defaultMap == MOORE_REG_INVALID_HANDLE) {
    b->defaultMap = handle;
  }

  ralTrace("Created register map '%s' in block '%s': base=0x%llx, bytes=%d "
           "(handle=%lld)",
           mapName.c_str(), b->name.c_str(), (unsigned long long)baseAddr,
           nBytes, (long long)handle);

  return handle;
}

extern "C" void __moore_reg_map_destroy(MooreRegMapHandle map) {
  auto &registry = getRegMapRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);

  if (map >= 0 && static_cast<size_t>(map) < registry.maps.size()) {
    ralTrace("Destroyed register map handle=%lld", (long long)map);
    registry.maps[map].reset();
  }
}

extern "C" MooreString __moore_reg_map_get_name(MooreRegMapHandle map) {
  RegMap *m = getRegMapRegistry().getMap(map);
  if (!m) {
    return {nullptr, 0};
  }

  char *data = static_cast<char *>(std::malloc(m->name.size()));
  if (!data)
    return {nullptr, 0};

  std::memcpy(data, m->name.data(), m->name.size());
  return {data, static_cast<int64_t>(m->name.size())};
}

extern "C" uint64_t __moore_reg_map_get_base_addr(MooreRegMapHandle map) {
  RegMap *m = getRegMapRegistry().getMap(map);
  return m ? m->baseAddr : 0;
}

extern "C" void __moore_reg_map_add_reg(MooreRegMapHandle map,
                                        MooreRegHandle reg, uint64_t offset,
                                        const char *rights) {
  RegMap *m = getRegMapRegistry().getMap(map);
  Register *r = getRegRegistry().getRegister(reg);

  if (!m || !r)
    return;

  std::string rightsStr = rights ? rights : "RW";
  m->entries.emplace_back(reg, offset, rightsStr);

  // Record mapping in register
  r->mapOffsets[map] = offset;

  ralTrace("Added register '%s' to map '%s' at offset 0x%llx (rights=%s)",
           r->name.c_str(), m->name.c_str(), (unsigned long long)offset,
           rightsStr.c_str());
}

extern "C" void __moore_reg_map_add_submap(MooreRegMapHandle parent,
                                           MooreRegMapHandle child,
                                           uint64_t offset) {
  RegMap *p = getRegMapRegistry().getMap(parent);
  RegMap *c = getRegMapRegistry().getMap(child);

  if (!p || !c)
    return;

  p->submaps.emplace_back(offset, child);

  ralTrace("Added sub-map '%s' to map '%s' at offset 0x%llx",
           c->name.c_str(), p->name.c_str(), (unsigned long long)offset);
}

extern "C" MooreRegHandle __moore_reg_map_get_reg_by_addr(MooreRegMapHandle map,
                                                          uint64_t addr) {
  RegMap *m = getRegMapRegistry().getMap(map);
  if (!m)
    return MOORE_REG_INVALID_HANDLE;

  // Calculate offset from base
  if (addr < m->baseAddr)
    return MOORE_REG_INVALID_HANDLE;

  uint64_t offset = addr - m->baseAddr;

  // Search in this map's entries
  for (const auto &entry : m->entries) {
    if (entry.offset == offset)
      return entry.reg;
  }

  // Search in sub-maps
  for (const auto &submap : m->submaps) {
    RegMap *sub = getRegMapRegistry().getMap(submap.second);
    if (sub && addr >= m->baseAddr + submap.first) {
      uint64_t subAddr = addr - submap.first;
      MooreRegHandle found = __moore_reg_map_get_reg_by_addr(submap.second,
                                                             subAddr);
      if (found != MOORE_REG_INVALID_HANDLE)
        return found;
    }
  }

  return MOORE_REG_INVALID_HANDLE;
}

extern "C" uint64_t __moore_reg_map_get_reg_offset(MooreRegMapHandle map,
                                                   MooreRegHandle reg) {
  RegMap *m = getRegMapRegistry().getMap(map);
  if (!m)
    return 0;

  for (const auto &entry : m->entries) {
    if (entry.reg == reg)
      return entry.offset;
  }

  return 0;
}

extern "C" void __moore_reg_map_set_sequencer(MooreRegMapHandle map,
                                              int64_t sequencer) {
  RegMap *m = getRegMapRegistry().getMap(map);
  if (m) {
    m->sequencer = sequencer;
    ralTrace("Set sequencer for map '%s' to %lld",
             m->name.c_str(), (long long)sequencer);
  }
}

extern "C" void __moore_reg_map_set_adapter(MooreRegMapHandle map,
                                            int64_t adapter) {
  RegMap *m = getRegMapRegistry().getMap(map);
  if (m) {
    m->adapter = adapter;
    ralTrace("Set adapter for map '%s' to %lld",
             m->name.c_str(), (long long)adapter);
  }
}

//===----------------------------------------------------------------------===//
// RAL Debugging and Tracing
//===----------------------------------------------------------------------===//

extern "C" void __moore_reg_set_trace_enabled(int32_t enable) {
  ralTraceEnabled = (enable != 0);
}

extern "C" int32_t __moore_reg_is_trace_enabled(void) {
  return ralTraceEnabled ? 1 : 0;
}

extern "C" void __moore_reg_block_print(MooreRegBlockHandle block) {
  RegBlock *b = getRegBlockRegistry().getBlock(block);
  if (!b) {
    std::printf("Invalid block handle: %lld\n", (long long)block);
    return;
  }

  std::printf("\n=== Register Block: %s ===\n", b->name.c_str());
  std::printf("  Locked: %s\n", b->locked ? "yes" : "no");
  std::printf("  Default Map: %lld\n", (long long)b->defaultMap);
  std::printf("  Registers: %zu\n", b->registers.size());

  for (const auto &entry : b->registers) {
    Register *r = getRegRegistry().getRegister(entry.reg);
    if (r) {
      std::printf("    - %s (offset=0x%llx, bits=%d, mirror=0x%llx)\n",
                  r->name.c_str(), (unsigned long long)entry.offset,
                  r->numBits, (unsigned long long)r->mirrorValue);

      for (size_t i = 0; i < r->fields.size(); ++i) {
        const auto &f = r->fields[i];
        std::printf("      . %s [%d:%d] (access=%d, reset=0x%llx)\n",
                    f->name.c_str(), f->lsbPos + f->numBits - 1, f->lsbPos,
                    (int)f->access, (unsigned long long)f->resetValue);
      }
    }
  }

  std::printf("  Sub-blocks: %zu\n", b->subBlocks.size());
  for (const auto &entry : b->subBlocks) {
    RegBlock *sub = getRegBlockRegistry().getBlock(entry.block);
    if (sub) {
      std::printf("    - %s (offset=0x%llx)\n",
                  sub->name.c_str(), (unsigned long long)entry.offset);
    }
  }

  std::printf("  Maps: %zu\n", b->maps.size());
  for (auto mapHandle : b->maps) {
    RegMap *m = getRegMapRegistry().getMap(mapHandle);
    if (m) {
      std::printf("    - %s (base=0x%llx, bytes=%d)%s\n",
                  m->name.c_str(), (unsigned long long)m->baseAddr, m->nBytes,
                  mapHandle == b->defaultMap ? " [default]" : "");
    }
  }

  std::printf("==============================\n");
}

extern "C" void __moore_reg_get_statistics(int64_t *totalRegs,
                                           int64_t *totalReads,
                                           int64_t *totalWrites) {
  if (totalRegs)
    *totalRegs = ralTotalRegs.load();
  if (totalReads)
    *totalReads = ralTotalReads.load();
  if (totalWrites)
    *totalWrites = ralTotalWrites.load();
}

extern "C" void __moore_reg_clear_all(void) {
  {
    auto &registry = getRegRegistry();
    std::lock_guard<std::mutex> lock(registry.regMutex);
    registry.registers.clear();
  }
  {
    auto &registry = getRegBlockRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.blocks.clear();
  }
  {
    auto &registry = getRegMapRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.maps.clear();
  }

  ralTotalRegs.store(0);
  ralTotalReads.store(0);
  ralTotalWrites.store(0);

  ralTrace("Cleared all RAL components");
}

//===----------------------------------------------------------------------===//
// UVM Message Reporting Infrastructure
//===----------------------------------------------------------------------===//
//
// This section implements UVM-compatible message reporting with verbosity
// filtering, severity tracking, and formatted output.
//

namespace {

/// State for UVM message reporting.
struct UvmReportState {
  /// Global verbosity threshold (default: UVM_MEDIUM = 200)
  std::atomic<int32_t> verbosity{200};

  /// Message counts by severity
  std::atomic<int32_t> infoCount{0};
  std::atomic<int32_t> warningCount{0};
  std::atomic<int32_t> errorCount{0};
  std::atomic<int32_t> fatalCount{0};

  /// Maximum error count before quit (0 = unlimited)
  std::atomic<int32_t> maxQuitCount{0};

  /// Actions by severity
  std::atomic<int32_t> infoAction{MOORE_UVM_DISPLAY};
  std::atomic<int32_t> warningAction{MOORE_UVM_DISPLAY | MOORE_UVM_COUNT};
  std::atomic<int32_t> errorAction{MOORE_UVM_DISPLAY | MOORE_UVM_COUNT};
  std::atomic<int32_t> fatalAction{MOORE_UVM_DISPLAY | MOORE_UVM_EXIT};

  /// Whether fatal should actually exit (for testing)
  std::atomic<bool> fatalExits{true};

  /// Current simulation time
  std::atomic<uint64_t> simTime{0};

  /// Per-ID verbosity overrides
  std::mutex idVerbosityMutex;
  std::unordered_map<std::string, int32_t> idVerbosity;
};

/// Get the singleton UVM report state.
UvmReportState &getUvmReportState() {
  static UvmReportState state;
  return state;
}

/// Format and print a UVM message.
/// Format: UVM_<SEVERITY> <filename>(<line>) @ <time>: <id> [<context>] <message>
void printUvmMessage(FILE *stream, const char *severity, const char *id,
                     int64_t idLen, const char *message, int64_t messageLen,
                     const char *filename, int64_t filenameLen, int32_t line,
                     const char *context, int64_t contextLen) {
  auto &state = getUvmReportState();

  // Print severity
  std::fprintf(stream, "%s ", severity);

  // Print filename and line if available
  if (filename && filenameLen > 0) {
    std::fwrite(filename, 1, static_cast<size_t>(filenameLen), stream);
    std::fprintf(stream, "(%d) ", line);
  }

  // Print timestamp
  std::fprintf(stream, "@ %llu: ", (unsigned long long)state.simTime.load());

  // Print ID
  if (id && idLen > 0) {
    std::fwrite(id, 1, static_cast<size_t>(idLen), stream);
    std::fputc(' ', stream);
  }

  // Print context if available
  if (context && contextLen > 0) {
    std::fputc('[', stream);
    std::fwrite(context, 1, static_cast<size_t>(contextLen), stream);
    std::fprintf(stream, "] ");
  }

  // Print message
  if (message && messageLen > 0) {
    std::fwrite(message, 1, static_cast<size_t>(messageLen), stream);
  }

  std::fputc('\n', stream);
  std::fflush(stream);
}

/// Get the verbosity threshold for a specific ID.
/// Returns the ID-specific verbosity if set, otherwise the global verbosity.
int32_t getIdVerbosity(const char *id, int64_t idLen) {
  auto &state = getUvmReportState();

  if (id && idLen > 0) {
    std::lock_guard<std::mutex> lock(state.idVerbosityMutex);
    std::string idStr(id, static_cast<size_t>(idLen));
    auto it = state.idVerbosity.find(idStr);
    if (it != state.idVerbosity.end()) {
      return it->second;
    }
  }

  return state.verbosity.load();
}

} // namespace

extern "C" void __moore_uvm_set_report_verbosity(int32_t verbosity) {
  getUvmReportState().verbosity.store(verbosity);
}

extern "C" int32_t __moore_uvm_get_report_verbosity(void) {
  return getUvmReportState().verbosity.load();
}

extern "C" void __moore_uvm_report_info(const char *id, int64_t idLen,
                                        const char *message, int64_t messageLen,
                                        int32_t verbosity, const char *filename,
                                        int64_t filenameLen, int32_t line,
                                        const char *context,
                                        int64_t contextLen) {
  auto &state = getUvmReportState();

  // Check verbosity filter
  int32_t threshold = getIdVerbosity(id, idLen);
  if (verbosity > threshold) {
    return; // Message filtered out
  }

  int32_t action = state.infoAction.load();

  // Display action
  if (action & MOORE_UVM_DISPLAY) {
    printUvmMessage(stdout, "UVM_INFO", id, idLen, message, messageLen,
                    filename, filenameLen, line, context, contextLen);
  }

  // Count action
  if (action & MOORE_UVM_COUNT) {
    state.infoCount.fetch_add(1);
  }
}

extern "C" void __moore_uvm_report_warning(const char *id, int64_t idLen,
                                           const char *message,
                                           int64_t messageLen, int32_t verbosity,
                                           const char *filename,
                                           int64_t filenameLen, int32_t line,
                                           const char *context,
                                           int64_t contextLen) {
  (void)verbosity; // Warnings are always displayed
  auto &state = getUvmReportState();

  int32_t action = state.warningAction.load();

  // Display action
  if (action & MOORE_UVM_DISPLAY) {
    printUvmMessage(stderr, "UVM_WARNING", id, idLen, message, messageLen,
                    filename, filenameLen, line, context, contextLen);
  }

  // Count action
  if (action & MOORE_UVM_COUNT) {
    state.warningCount.fetch_add(1);
  }
}

extern "C" void __moore_uvm_report_error(const char *id, int64_t idLen,
                                         const char *message, int64_t messageLen,
                                         int32_t verbosity, const char *filename,
                                         int64_t filenameLen, int32_t line,
                                         const char *context,
                                         int64_t contextLen) {
  (void)verbosity; // Errors are always displayed
  auto &state = getUvmReportState();

  int32_t action = state.errorAction.load();

  // Display action
  if (action & MOORE_UVM_DISPLAY) {
    printUvmMessage(stderr, "UVM_ERROR", id, idLen, message, messageLen,
                    filename, filenameLen, line, context, contextLen);
  }

  // Count action
  if (action & MOORE_UVM_COUNT) {
    int32_t newCount = state.errorCount.fetch_add(1) + 1;

    // Check max quit count
    int32_t maxQuit = state.maxQuitCount.load();
    if (maxQuit > 0 && newCount >= maxQuit) {
      std::fprintf(stderr, "UVM_ERROR: Quit count reached (%d errors)\n",
                   newCount);

      if (action & MOORE_UVM_EXIT) {
        if (state.fatalExits.load()) {
          std::exit(1);
        }
      }
    }
  }
}

extern "C" void __moore_uvm_report_fatal(const char *id, int64_t idLen,
                                         const char *message, int64_t messageLen,
                                         int32_t verbosity, const char *filename,
                                         int64_t filenameLen, int32_t line,
                                         const char *context,
                                         int64_t contextLen) {
  (void)verbosity; // Fatal messages are always displayed
  auto &state = getUvmReportState();

  int32_t action = state.fatalAction.load();

  // Display action
  if (action & MOORE_UVM_DISPLAY) {
    printUvmMessage(stderr, "UVM_FATAL", id, idLen, message, messageLen,
                    filename, filenameLen, line, context, contextLen);
  }

  // Count action (always count fatal)
  state.fatalCount.fetch_add(1);

  // Exit action
  if (action & MOORE_UVM_EXIT) {
    if (state.fatalExits.load()) {
      std::exit(1);
    }
  }
}

extern "C" int32_t __moore_uvm_report_enabled(int32_t verbosity,
                                              int32_t severity, const char *id,
                                              int64_t idLen) {
  // Warnings, errors, and fatals are always enabled
  if (severity != MOORE_UVM_INFO) {
    return 1;
  }

  // For info messages, check verbosity
  int32_t threshold = getIdVerbosity(id, idLen);
  return verbosity <= threshold ? 1 : 0;
}

extern "C" void __moore_uvm_set_report_id_verbosity(const char *id,
                                                    int64_t idLen,
                                                    int32_t verbosity) {
  if (!id || idLen <= 0) {
    return;
  }

  auto &state = getUvmReportState();
  std::lock_guard<std::mutex> lock(state.idVerbosityMutex);
  std::string idStr(id, static_cast<size_t>(idLen));
  state.idVerbosity[idStr] = verbosity;
}

extern "C" int32_t __moore_uvm_get_report_count(int32_t severity) {
  auto &state = getUvmReportState();

  switch (severity) {
  case MOORE_UVM_INFO:
    return state.infoCount.load();
  case MOORE_UVM_WARNING:
    return state.warningCount.load();
  case MOORE_UVM_ERROR:
    return state.errorCount.load();
  case MOORE_UVM_FATAL:
    return state.fatalCount.load();
  default:
    return 0;
  }
}

extern "C" void __moore_uvm_reset_report_counts(void) {
  auto &state = getUvmReportState();
  state.infoCount.store(0);
  state.warningCount.store(0);
  state.errorCount.store(0);
  state.fatalCount.store(0);
}

extern "C" void __moore_uvm_set_max_quit_count(int32_t count) {
  getUvmReportState().maxQuitCount.store(count);
}

extern "C" int32_t __moore_uvm_get_max_quit_count(void) {
  return getUvmReportState().maxQuitCount.load();
}

extern "C" void __moore_uvm_set_report_severity_action(int32_t severity,
                                                       int32_t action) {
  auto &state = getUvmReportState();

  switch (severity) {
  case MOORE_UVM_INFO:
    state.infoAction.store(action);
    break;
  case MOORE_UVM_WARNING:
    state.warningAction.store(action);
    break;
  case MOORE_UVM_ERROR:
    state.errorAction.store(action);
    break;
  case MOORE_UVM_FATAL:
    state.fatalAction.store(action);
    break;
  }
}

extern "C" int32_t __moore_uvm_get_report_severity_action(int32_t severity) {
  auto &state = getUvmReportState();

  switch (severity) {
  case MOORE_UVM_INFO:
    return state.infoAction.load();
  case MOORE_UVM_WARNING:
    return state.warningAction.load();
  case MOORE_UVM_ERROR:
    return state.errorAction.load();
  case MOORE_UVM_FATAL:
    return state.fatalAction.load();
  default:
    return 0;
  }
}

extern "C" void __moore_uvm_report_summarize(void) {
  auto &state = getUvmReportState();

  int32_t infoCount = state.infoCount.load();
  int32_t warningCount = state.warningCount.load();
  int32_t errorCount = state.errorCount.load();
  int32_t fatalCount = state.fatalCount.load();

  std::fprintf(stdout, "\n--- UVM Report Summary ---\n");
  std::fprintf(stdout, "** Report counts by severity **\n");
  std::fprintf(stdout, "  UVM_INFO:    %d\n", infoCount);
  std::fprintf(stdout, "  UVM_WARNING: %d\n", warningCount);
  std::fprintf(stdout, "  UVM_ERROR:   %d\n", errorCount);
  std::fprintf(stdout, "  UVM_FATAL:   %d\n", fatalCount);

  if (errorCount > 0 || fatalCount > 0) {
    std::fprintf(stdout, "\n** SIMULATION FAILED **\n");
  } else if (warningCount > 0) {
    std::fprintf(stdout, "\n** SIMULATION PASSED (with warnings) **\n");
  } else {
    std::fprintf(stdout, "\n** SIMULATION PASSED **\n");
  }

  std::fflush(stdout);
}

extern "C" void __moore_uvm_set_fatal_exits(bool should_exit) {
  getUvmReportState().fatalExits.store(should_exit);
}

extern "C" uint64_t __moore_uvm_get_time(void) {
  return getUvmReportState().simTime.load();
}

extern "C" void __moore_uvm_set_time(uint64_t time) {
  getUvmReportState().simTime.store(time);
}

//===----------------------------------------------------------------------===//
// SystemVerilog Semaphore Support
//===----------------------------------------------------------------------===//

namespace {

/// Internal semaphore structure.
struct Semaphore {
  std::mutex mutex;
  std::condition_variable cv;
  int32_t keyCount;

  explicit Semaphore(int32_t initialKeys) : keyCount(initialKeys) {}
};

/// Registry for semaphores.
struct SemaphoreRegistry {
  std::mutex mutex;
  std::unordered_map<MooreSemaphoreHandle, std::unique_ptr<Semaphore>> semaphores;
  MooreSemaphoreHandle nextHandle = 1;

  MooreSemaphoreHandle create(int32_t keyCount) {
    std::lock_guard<std::mutex> lock(mutex);
    MooreSemaphoreHandle handle = nextHandle++;
    semaphores[handle] = std::make_unique<Semaphore>(keyCount);
    return handle;
  }

  void destroy(MooreSemaphoreHandle handle) {
    std::lock_guard<std::mutex> lock(mutex);
    semaphores.erase(handle);
  }

  Semaphore *get(MooreSemaphoreHandle handle) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = semaphores.find(handle);
    if (it != semaphores.end()) {
      return it->second.get();
    }
    return nullptr;
  }
};

SemaphoreRegistry &getSemaphoreRegistry() {
  static SemaphoreRegistry registry;
  return registry;
}

} // namespace

extern "C" MooreSemaphoreHandle __moore_semaphore_create(int32_t keyCount) {
  if (keyCount < 0) {
    return MOORE_SEMAPHORE_INVALID_HANDLE;
  }
  return getSemaphoreRegistry().create(keyCount);
}

extern "C" void __moore_semaphore_destroy(MooreSemaphoreHandle sem) {
  if (sem == MOORE_SEMAPHORE_INVALID_HANDLE) {
    return;
  }
  getSemaphoreRegistry().destroy(sem);
}

extern "C" void __moore_semaphore_put(MooreSemaphoreHandle sem,
                                      int32_t keyCount) {
  if (sem == MOORE_SEMAPHORE_INVALID_HANDLE || keyCount <= 0) {
    return;
  }

  auto *semaphore = getSemaphoreRegistry().get(sem);
  if (!semaphore) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(semaphore->mutex);
    semaphore->keyCount += keyCount;
  }
  // Notify all waiting threads so they can check if enough keys are available
  semaphore->cv.notify_all();
}

extern "C" void __moore_semaphore_get(MooreSemaphoreHandle sem,
                                      int32_t keyCount) {
  if (sem == MOORE_SEMAPHORE_INVALID_HANDLE || keyCount <= 0) {
    return;
  }

  auto *semaphore = getSemaphoreRegistry().get(sem);
  if (!semaphore) {
    return;
  }

  std::unique_lock<std::mutex> lock(semaphore->mutex);
  semaphore->cv.wait(lock,
                     [&semaphore, keyCount]() {
                       return semaphore->keyCount >= keyCount;
                     });
  semaphore->keyCount -= keyCount;
}

extern "C" int32_t __moore_semaphore_try_get(MooreSemaphoreHandle sem,
                                             int32_t keyCount) {
  if (sem == MOORE_SEMAPHORE_INVALID_HANDLE || keyCount <= 0) {
    return 0;
  }

  auto *semaphore = getSemaphoreRegistry().get(sem);
  if (!semaphore) {
    return 0;
  }

  std::lock_guard<std::mutex> lock(semaphore->mutex);
  if (semaphore->keyCount >= keyCount) {
    semaphore->keyCount -= keyCount;
    return 1;
  }
  return 0;
}

extern "C" int32_t __moore_semaphore_get_key_count(MooreSemaphoreHandle sem) {
  if (sem == MOORE_SEMAPHORE_INVALID_HANDLE) {
    return 0;
  }

  auto *semaphore = getSemaphoreRegistry().get(sem);
  if (!semaphore) {
    return 0;
  }

  std::lock_guard<std::mutex> lock(semaphore->mutex);
  return semaphore->keyCount;
}

//===----------------------------------------------------------------------===//
// UVM Root Re-entrancy Support
//===----------------------------------------------------------------------===//
//
// The UVM library has a re-entrancy issue in the uvm_root singleton pattern:
//
// 1. m_uvm_get_root() checks if m_inst == null
// 2. If null, it calls new uvm_root()
// 3. uvm_root::new() sets m_inst = this BEFORE returning
// 4. uvm_root::new() calls uvm_component::new()
// 5. uvm_component::new() calls cs.get_root() which calls m_uvm_get_root()
// 6. m_uvm_get_root() sees m_inst != null, so returns m_inst
// 7. Then it checks if m_inst != uvm_top and reports an error
//
// The problem is that uvm_top is not set until AFTER uvm_root::new() returns,
// but m_inst is set at the START of uvm_root::new(). So during the re-entrant
// call, m_inst != uvm_top always, causing a false warning.
//
// Our fix: track when root construction is in progress and skip the check.
//

namespace {

/// State for UVM root re-entrancy tracking.
struct UvmRootConstructionState {
  /// Flag indicating root construction is in progress.
  std::atomic<bool> constructing{false};

  /// The root instance being constructed (or null).
  std::atomic<void *> rootInst{nullptr};
};

UvmRootConstructionState &getUvmRootState() {
  static UvmRootConstructionState state;
  return state;
}

} // anonymous namespace

extern "C" void __moore_uvm_root_constructing_start(void) {
  getUvmRootState().constructing.store(true);
}

extern "C" void __moore_uvm_root_constructing_end(void) {
  getUvmRootState().constructing.store(false);
}

extern "C" bool __moore_uvm_is_root_constructing(void) {
  return getUvmRootState().constructing.load();
}

extern "C" void __moore_uvm_set_root_inst(void *inst) {
  getUvmRootState().rootInst.store(inst);
}

extern "C" void *__moore_uvm_get_root_inst(void) {
  return getUvmRootState().rootInst.load();
}
