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
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <regex>
#include <unordered_map>
#include <string>
#include <vector>

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

/// Type tag stored at the beginning of the allocation to identify the array
/// type.
enum class AssocArrayType : int32_t { StringKey = 0, IntKey = 1 };

/// Wrapper that holds the type tag and pointer to the actual array.
struct AssocArrayHeader {
  AssocArrayType type;
  void *array;
};

} // anonymous namespace

extern "C" void *__moore_assoc_create(int32_t key_size, int32_t value_size) {
  auto *header = new AssocArrayHeader;
  if (key_size == 0) {
    // String-keyed associative array
    auto *arr = new StringKeyAssocArray;
    arr->valueSize = value_size;
    header->type = AssocArrayType::StringKey;
    header->array = arr;
  } else {
    // Integer-keyed associative array
    auto *arr = new IntKeyAssocArray;
    arr->keySize = key_size;
    arr->valueSize = value_size;
    header->type = AssocArrayType::IntKey;
    header->array = arr;
  }
  return header;
}

extern "C" int64_t __moore_assoc_size(void *array) {
  if (!array)
    return 0;
  auto *header = static_cast<AssocArrayHeader *>(array);
  if (header->type == AssocArrayType::StringKey) {
    auto *arr = static_cast<StringKeyAssocArray *>(header->array);
    return static_cast<int64_t>(arr->data.size());
  } else {
    auto *arr = static_cast<IntKeyAssocArray *>(header->array);
    return static_cast<int64_t>(arr->data.size());
  }
}

extern "C" void __moore_assoc_delete(void *array) {
  if (!array)
    return;
  auto *header = static_cast<AssocArrayHeader *>(array);
  if (header->type == AssocArrayType::StringKey) {
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
  if (header->type == AssocArrayType::StringKey) {
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
  if (header->type == AssocArrayType::StringKey) {
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
  if (header->type == AssocArrayType::StringKey) {
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
  if (header->type == AssocArrayType::StringKey) {
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
  if (header->type == AssocArrayType::StringKey) {
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

extern "C" void *__moore_assoc_get_ref(void *array, void *key,
                                       int32_t value_size) {
  if (!array || !key)
    return nullptr;
  auto *header = static_cast<AssocArrayHeader *>(array);
  if (header->type == AssocArrayType::StringKey) {
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

  constexpr int64_t kMaxCycleBits = 12;
  if (bitWidth > kMaxCycleBits) {
    uint64_t value = static_cast<uint64_t>(__moore_urandom());
    value |= static_cast<uint64_t>(__moore_urandom()) << 32;
    if (bitWidth < 64)
      value &= ((1ULL << bitWidth) - 1);
    return static_cast<int64_t>(value);
  }

  const uint64_t maxValue = (1ULL << bitWidth) - 1;
  struct RandCState {
    std::vector<uint64_t> remaining;
  };

  static std::mutex randcMutex;
  static std::unordered_map<void *, RandCState> randcStates;

  std::lock_guard<std::mutex> lock(randcMutex);
  auto &state = randcStates[fieldPtr];
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

/// Helper to track unique values seen by a coverpoint using a simple set.
/// For production use, this could be replaced with a more efficient data
/// structure like a hash set or bit vector.
struct CoverpointTracker {
  std::map<int64_t, int64_t> valueCounts; // value -> hit count
};

/// Map from coverpoint to its value tracker.
/// This is separate from the MooreCoverpoint struct to keep the C API simple.
thread_local std::map<MooreCoverpoint *, CoverpointTracker> coverpointTrackers;

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

  // Initialize the covergroup
  cg->name = name;
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
  cp->name = name;
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

      // Free bin array if present
      if (covergroup->coverpoints[i]->bins) {
        std::free(covergroup->coverpoints[i]->bins);
      }
      std::free(covergroup->coverpoints[i]);
    }
  }

  // Free the coverpoints array
  if (covergroup->coverpoints) {
    std::free(covergroup->coverpoints);
  }

  // Free the covergroup itself
  std::free(covergroup);
}

// Forward declaration for explicit bin update helper
namespace {
void updateExplicitBinsHelper(MooreCoverpoint *cp, int64_t value);
} // namespace

extern "C" void __moore_coverpoint_sample(void *cg, int32_t cp_index,
                                           int64_t value) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp)
    return;

  // Update hit count
  cp->hits++;

  // Update min/max tracking
  if (value < cp->min_val)
    cp->min_val = value;
  if (value > cp->max_val)
    cp->max_val = value;

  // Track unique values
  auto trackerIt = coverpointTrackers.find(cp);
  if (trackerIt != coverpointTrackers.end()) {
    trackerIt->second.valueCounts[value]++;
  }

  // Update explicit bins if present
  updateExplicitBinsHelper(cp, value);
}

extern "C" double __moore_coverpoint_get_coverage(void *cg, int32_t cp_index) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || cp_index < 0 || cp_index >= covergroup->num_coverpoints)
    return 0.0;

  auto *cp = covergroup->coverpoints[cp_index];
  if (!cp || cp->hits == 0)
    return 0.0;

  // For auto bins, calculate coverage as the ratio of unique values seen
  // to the theoretical range. Since we don't know the type's range at runtime,
  // we use the actual range of values seen plus some margin.
  auto trackerIt = coverpointTrackers.find(cp);
  if (trackerIt == coverpointTrackers.end())
    return 0.0;

  int64_t uniqueValues = static_cast<int64_t>(trackerIt->second.valueCounts.size());

  // If we have explicit bins, use those
  if (cp->bins && cp->num_bins > 0) {
    int64_t hitBins = 0;
    for (int32_t i = 0; i < cp->num_bins; ++i) {
      if (cp->bins[i] > 0)
        hitBins++;
    }
    return (100.0 * hitBins) / cp->num_bins;
  }

  // For auto bins, estimate coverage based on unique values seen.
  // We assume the goal is to cover the range [min_val, max_val].
  // If the range is 0 (single value), coverage is 100%.
  if (cp->min_val > cp->max_val)
    return 0.0; // No valid samples

  int64_t range = cp->max_val - cp->min_val + 1;
  if (range <= 0)
    return 100.0; // Single value = 100% coverage

  // Calculate coverage percentage
  // Cap at 100% since unique values might exceed expected range
  double coverage = (100.0 * uniqueValues) / range;
  return coverage > 100.0 ? 100.0 : coverage;
}

extern "C" double __moore_covergroup_get_coverage(void *cg) {
  auto *covergroup = static_cast<MooreCovergroup *>(cg);
  if (!covergroup || covergroup->num_coverpoints == 0)
    return 0.0;

  // Calculate average coverage across all coverpoints
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

namespace {

/// Structure to store explicit bin data for a coverpoint.
/// Stored separately to maintain backward compatibility with the C API.
struct ExplicitBinData {
  std::vector<MooreCoverageBin> bins;
};

/// Map from coverpoint to its explicit bin data.
thread_local std::map<MooreCoverpoint *, ExplicitBinData> explicitBinData;

} // anonymous namespace

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
  cp->name = name;
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
    // Wildcard matching is future work
    return false;
  case MOORE_BIN_TRANSITION:
    // Transition matching requires state tracking
    return false;
  default:
    return false;
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
} // namespace

extern "C" int32_t uvm_hdl_check_path(MooreString *path) {
  std::string key;
  if (!getPathKey(path, key))
    return 0;
  return 1;
}

extern "C" int32_t uvm_hdl_deposit(MooreString *path, uvm_hdl_data_t value) {
  std::string key;
  if (!getPathKey(path, key))
    return 0;
  std::lock_guard<std::mutex> lock(hdlMutex);
  auto &entry = hdlValues[key];
  entry.value = value;
  return 1;
}

extern "C" int32_t uvm_hdl_force(MooreString *path, uvm_hdl_data_t value) {
  std::string key;
  if (!getPathKey(path, key))
    return 0;
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
// Simple regex stub: just stores the pattern string for later matching.
// A full implementation would use a regex library like PCRE2 that doesn't
// require exception support.
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

  // Create a stub regex object
  // Note: std::regex compilation can throw exceptions, but we build
  // with -fno-exceptions. For now, use a simple substring match stub.
  // A full implementation would use a regex library that doesn't need exceptions.
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

  // Simplified matching: just check if pattern is a substring
  // A real implementation would use proper regex matching (e.g., PCRE2 library)
  size_t pos = target.find(stub->pattern);
  if (pos != std::string::npos) {
    lastMatchBuffer = stub->pattern; // Store match for uvm_re_buffer
    return static_cast<int32_t>(pos); // Return match position
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
bool cmdLineArgsInitialized = false;

void initCommandLineArgs() {
  if (cmdLineArgsInitialized)
    return;
  cmdLineArgsInitialized = true;

  const char *env = std::getenv("CIRCT_UVM_ARGS");
  if (!env)
    env = std::getenv("UVM_ARGS");
  if (!env)
    return;

  std::string args(env);
  size_t i = 0;
  while (i < args.size()) {
    while (i < args.size() &&
           std::isspace(static_cast<unsigned char>(args[i])))
      ++i;
    if (i >= args.size())
      break;
    size_t start = i;
    while (i < args.size() &&
           !std::isspace(static_cast<unsigned char>(args[i])))
      ++i;
    if (i > start)
      cmdLineArgs.emplace_back(args.substr(start, i - start));
  }
}
} // namespace

extern "C" MooreString uvm_dpi_get_next_arg_c(int32_t *idx) {
  initCommandLineArgs();

  if (!idx) {
    MooreString result = {nullptr, 0};
    return result;
  }

  // Stub: Return empty string (no arguments)
  // A real implementation would parse actual command line arguments
  if (*idx >= static_cast<int32_t>(cmdLineArgs.size())) {
    MooreString result = {nullptr, 0};
    return result;
  }

  const std::string &arg = cmdLineArgs[*idx];
  (*idx)++;

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
