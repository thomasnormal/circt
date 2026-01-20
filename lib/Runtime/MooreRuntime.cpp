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
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <regex>
#include <set>
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

      // Remove from options map
      coverpointOptions.erase(covergroup->coverpoints[i]);

      // Remove from explicit bin data map
      explicitBinData.erase(covergroup->coverpoints[i]);

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

  // Clean up cross coverage data (implemented later in file)
  extern void __moore_cross_cleanup_for_covergroup(MooreCovergroup *);
  __moore_cross_cleanup_for_covergroup(covergroup);

  // Clean up covergroup options
  covergroupOptions.erase(covergroup);

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
  auto trackerIt = coverpointTrackers.find(cp);
  int64_t prevValue = 0;
  bool hasPrev = false;
  if (trackerIt != coverpointTrackers.end()) {
    prevValue = trackerIt->second.prevValue;
    hasPrev = trackerIt->second.hasPrevValue;
    trackerIt->second.valueCounts[value]++;
  }

  // Update explicit bins if present
  updateExplicitBinsHelper(cp, value);

  // Update transition bins if present
  updateTransitionBinsHelper(cp, value, prevValue, hasPrev);

  // Update previous value for transition tracking
  if (trackerIt != coverpointTrackers.end()) {
    trackerIt->second.prevValue = value;
    trackerIt->second.hasPrevValue = true;
  }
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

    for (int32_t i = 0; i < cp->num_bins; ++i) {
      // Skip ignore bins - they don't count toward coverage
      if (binDataIt != explicitBinData.end() &&
          i < static_cast<int32_t>(binDataIt->second.bins.size()) &&
          binDataIt->second.bins[i].kind == MOORE_BIN_KIND_IGNORE) {
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
  for (const auto &kv : it->second.crossBins) {
    // Check if this bin belongs to this cross by comparing with first cp index
    if (kv.first.size() == static_cast<size_t>(cross.num_cps)) {
      binsHit++;
    }
  }

  if (totalPossibleBins == 0)
    return 0.0;

  double coverage = (100.0 * binsHit) / totalPossibleBins;
  return coverage > 100.0 ? 100.0 : coverage;
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
  html += "  </style>\n";
  html += "</head>\n";
  html += "<body>\n";
  html += "  <div class=\"container\">\n";
  html += "    <div class=\"header\">\n";
  html += "      <h1>CIRCT Coverage Report</h1>\n";
  html += "      <p class=\"meta\">Generated by circt-moore-runtime</p>\n";
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

    html += "    <div class=\"covergroup\">\n";
    html += "      <h2>" + std::string(cg->name ? cg->name : "(unnamed)") + "</h2>\n";
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
      html += "      <table>\n";
      html += "        <thead>\n";
      html += "          <tr>\n";
      html += "            <th>Coverpoint</th>\n";
      html += "            <th>Hits</th>\n";
      html += "            <th>Unique Values</th>\n";
      html += "            <th>Range</th>\n";
      html += "            <th>Coverage</th>\n";
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

        std::snprintf(buf, sizeof(buf), "%.1f%%", cpCoverage);
        std::string cpClass = cpCoverage >= 80 ? "coverage-high" :
                              (cpCoverage >= 50 ? "coverage-med" : "coverage-low");
        html += "            <td><span style=\"color: var(--" +
                (cpCoverage >= 80 ? std::string("success") :
                 (cpCoverage >= 50 ? std::string("warning") : std::string("danger"))) +
                ")\">" + std::string(buf) + "</span></td>\n";
        html += "          </tr>\n";
      }

      html += "        </tbody>\n";
      html += "      </table>\n";
    }

    html += "    </div>\n";
  }

  html += "  </div>\n";
  html += "</body>\n";
  html += "</html>\n";

  std::fwrite(html.c_str(), 1, html.size(), fp);
  std::fclose(fp);

  return 0;
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

  std::vector<CovergroupData> covergroups;

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

} // anonymous namespace

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
  std::lock_guard<std::mutex> lock(hdlMutex);
  (void)hdlValues[key];
  return 1;
}

extern "C" int32_t uvm_hdl_deposit(MooreString *path, uvm_hdl_data_t value) {
  std::string key;
  if (!getPathKey(path, key))
    return 0;
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
  std::lock_guard<std::mutex> lock(hdlMutex);
  auto &entry = hdlValues[handle->name];
  entry.value = *static_cast<uvm_hdl_data_t *>(value->value);
  entry.forced = (flags != 0);
  return 1;
}
