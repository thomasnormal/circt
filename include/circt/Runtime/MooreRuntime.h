//===- MooreRuntime.h - Runtime library for Moore dialect -------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the runtime library functions required by the MooreToCore
// lowering pass. These functions implement operations that cannot be lowered
// directly to LLVM IR, such as queue operations, string manipulation, dynamic
// arrays, and associative array iteration.
//
// The runtime is designed to be linked with the compiled simulation binary.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_RUNTIME_MOORERUNTIME_H
#define CIRCT_RUNTIME_MOORERUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// String Type
//===----------------------------------------------------------------------===//
//
// Strings are represented as a {ptr, len} pair. This matches the LLVM struct
// type used in MooreToCore: !llvm.struct<(ptr, i64)>
//

/// String structure used by the Moore runtime.
/// - data: pointer to the string contents (not null-terminated)
/// - len: length of the string in bytes
typedef struct {
  char *data;
  int64_t len;
} MooreString;

//===----------------------------------------------------------------------===//
// Queue/Dynamic Array Type
//===----------------------------------------------------------------------===//
//
// Queues and dynamic arrays are represented as a {ptr, len} pair.
// This matches !llvm.struct<(ptr, i64)> used in MooreToCore.
//

/// Queue/dynamic array structure.
/// - data: pointer to the element storage
/// - len: number of elements
typedef struct {
  void *data;
  int64_t len;
} MooreQueue;

//===----------------------------------------------------------------------===//
// Queue Operations
//===----------------------------------------------------------------------===//

/// Get the maximum element from a queue.
/// @param queue Pointer to the queue structure
/// @return A new queue containing only the maximum element
MooreQueue __moore_queue_max(MooreQueue *queue);

/// Get the minimum element from a queue.
/// @param queue Pointer to the queue structure
/// @return A new queue containing only the minimum element
MooreQueue __moore_queue_min(MooreQueue *queue);

/// Push an element to the back of a queue.
/// @param queue Pointer to the queue structure (modified in place)
/// @param element Pointer to the element to push
/// @param element_size Size of the element in bytes
void __moore_queue_push_back(MooreQueue *queue, void *element,
                             int64_t element_size);

/// Push an element to the front of a queue.
/// @param queue Pointer to the queue structure (modified in place)
/// @param element Pointer to the element to push
/// @param element_size Size of the element in bytes
void __moore_queue_push_front(MooreQueue *queue, void *element,
                              int64_t element_size);

/// Pop an element from the back of a queue.
/// @param queue Pointer to the queue structure (modified in place)
/// @param element_size Size of the element in bytes
/// @return The popped element value (as 64-bit integer)
int64_t __moore_queue_pop_back(MooreQueue *queue, int64_t element_size);

/// Pop an element from the front of a queue.
/// @param queue Pointer to the queue structure (modified in place)
/// @param element_size Size of the element in bytes
/// @return The popped element value (as 64-bit integer)
int64_t __moore_queue_pop_front(MooreQueue *queue, int64_t element_size);

//===----------------------------------------------------------------------===//
// Dynamic Array Operations
//===----------------------------------------------------------------------===//

/// Create a new dynamic array with the specified size.
/// Elements are zero-initialized.
/// @param size Number of elements to allocate
/// @return A new dynamic array structure
MooreQueue __moore_dyn_array_new(int32_t size);

/// Create a new dynamic array by copying from an existing array.
/// @param size Number of elements to allocate
/// @param init Pointer to the source array to copy from
/// @return A new dynamic array structure with copied contents
MooreQueue __moore_dyn_array_new_copy(int32_t size, void *init);

//===----------------------------------------------------------------------===//
// Associative Array Operations
//===----------------------------------------------------------------------===//

/// Delete all entries from an associative array.
/// @param array Pointer to the associative array
void __moore_assoc_delete(void *array);

/// Delete a specific key from an associative array.
/// @param array Pointer to the associative array
/// @param key Pointer to the key to delete
void __moore_assoc_delete_key(void *array, void *key);

/// Get the first key from an associative array.
/// @param array Pointer to the associative array
/// @param key_out Pointer where the first key will be stored
/// @return true if a key was found, false if the array is empty
bool __moore_assoc_first(void *array, void *key_out);

/// Get the next key from an associative array (after the given key).
/// @param array Pointer to the associative array
/// @param key_ref Pointer to current key; updated to next key on success
/// @return true if a next key was found, false otherwise
bool __moore_assoc_next(void *array, void *key_ref);

/// Get the last key from an associative array.
/// @param array Pointer to the associative array
/// @param key_out Pointer where the last key will be stored
/// @return true if a key was found, false if the array is empty
bool __moore_assoc_last(void *array, void *key_out);

/// Get the previous key from an associative array (before the given key).
/// @param array Pointer to the associative array
/// @param key_ref Pointer to current key; updated to previous key on success
/// @return true if a previous key was found, false otherwise
bool __moore_assoc_prev(void *array, void *key_ref);

//===----------------------------------------------------------------------===//
// String Operations
//===----------------------------------------------------------------------===//

/// Get the length of a string.
/// @param str Pointer to the string structure
/// @return Length of the string in characters
int32_t __moore_string_len(MooreString *str);

/// Convert a string to uppercase.
/// @param str Pointer to the input string
/// @return A new string with all characters converted to uppercase
MooreString __moore_string_toupper(MooreString *str);

/// Convert a string to lowercase.
/// @param str Pointer to the input string
/// @return A new string with all characters converted to lowercase
MooreString __moore_string_tolower(MooreString *str);

/// Get a character at a specific index.
/// @param str Pointer to the string
/// @param index Zero-based index of the character to retrieve
/// @return The character at the given index, or 0 if index is out of bounds
int8_t __moore_string_getc(MooreString *str, int32_t index);

/// Extract a substring.
/// @param str Pointer to the source string
/// @param start Starting index (0-based)
/// @param len Number of characters to extract
/// @return A new string containing the substring
MooreString __moore_string_substr(MooreString *str, int32_t start, int32_t len);

/// Convert an integer to its ASCII representation and assign to a string.
/// This implements the SystemVerilog .itoa() method.
/// @param value The integer value to convert
/// @return A new string containing the decimal representation
MooreString __moore_string_itoa(int64_t value);

/// Concatenate two strings.
/// @param lhs Pointer to the left string
/// @param rhs Pointer to the right string
/// @return A new string containing the concatenation
MooreString __moore_string_concat(MooreString *lhs, MooreString *rhs);

/// Compare two strings lexicographically.
/// @param lhs Pointer to the left string
/// @param rhs Pointer to the right string
/// @return < 0 if lhs < rhs, 0 if equal, > 0 if lhs > rhs
int32_t __moore_string_cmp(MooreString *lhs, MooreString *rhs);

/// Convert an integer to its string representation.
/// Similar to itoa but treats the value as unsigned for UVM compatibility.
/// @param value The integer value to convert
/// @return A new string containing the representation
MooreString __moore_int_to_string(int64_t value);

/// Convert a string to an integer.
/// Parses a decimal integer from the string.
/// @param str Pointer to the string to parse
/// @return The parsed integer value, or 0 if parsing fails
int64_t __moore_string_to_int(MooreString *str);

//===----------------------------------------------------------------------===//
// Streaming Concatenation Operations
//===----------------------------------------------------------------------===//

/// Concatenate all strings in a queue into a single string.
/// Used for streaming concatenation of string queues: {>>{string_queue}}
/// @param queue Pointer to the queue of strings
/// @param isRightToLeft If true, concatenate right-to-left; otherwise left-to-right
/// @return A new string containing the concatenation of all queue elements
MooreString __moore_stream_concat_strings(MooreQueue *queue, bool isRightToLeft);

/// Pack all elements of a queue/dynamic array into a single integer value.
/// Used for streaming concatenation of non-string types: {>>{int_queue}}
/// @param queue Pointer to the queue/array
/// @param elementBitWidth Bit width of each element
/// @param isRightToLeft If true, pack right-to-left; otherwise left-to-right
/// @return The packed bits as a 64-bit integer
int64_t __moore_stream_concat_bits(MooreQueue *queue, int32_t elementBitWidth,
                                   bool isRightToLeft);

//===----------------------------------------------------------------------===//
// Event Operations
//===----------------------------------------------------------------------===//

/// Check if an event was triggered in the current time slot.
/// Implements the SystemVerilog `.triggered` property on events.
/// @param event Pointer to the event (stored as a boolean flag)
/// @return true if the event was triggered, false otherwise
bool __moore_event_triggered(bool *event);

//===----------------------------------------------------------------------===//
// Simulation Control Operations
//===----------------------------------------------------------------------===//

/// Wait until a condition becomes true.
/// Implements the SystemVerilog `wait(condition)` statement.
/// This function suspends execution until the condition is non-zero.
/// @param condition The condition value to wait for (non-zero = true)
void __moore_wait_condition(int32_t condition);

//===----------------------------------------------------------------------===//
// Random Number Generation
//===----------------------------------------------------------------------===//

/// Generate a pseudo-random 32-bit unsigned integer.
/// Implements the SystemVerilog $urandom system function.
/// @return A 32-bit pseudo-random unsigned integer
uint32_t __moore_urandom(void);

/// Seed the pseudo-random number generator.
/// Implements the seeded form of $urandom: $urandom(seed).
/// @param seed The seed value for the random number generator
/// @return A 32-bit pseudo-random unsigned integer
uint32_t __moore_urandom_seeded(int32_t seed);

/// Generate a pseudo-random unsigned integer within a range.
/// Implements the SystemVerilog $urandom_range system function.
/// If min > max, the values are automatically swapped.
/// @param maxval The maximum value (inclusive)
/// @param minval The minimum value (inclusive), defaults to 0
/// @return A random value in the range [min(minval, maxval), max(minval, maxval)]
uint32_t __moore_urandom_range(uint32_t maxval, uint32_t minval);

/// Generate a true random 32-bit signed integer.
/// Implements the SystemVerilog $random system function.
/// @return A 32-bit random signed integer
int32_t __moore_random(void);

/// Seed the true random number generator and return a random value.
/// Implements the seeded form of $random: $random(seed).
/// @param seed The seed value for the random number generator
/// @return A 32-bit random signed integer
int32_t __moore_random_seeded(int32_t seed);

//===----------------------------------------------------------------------===//
// Memory Management
//===----------------------------------------------------------------------===//

/// Free memory allocated by the Moore runtime.
/// Should be called to release strings and arrays when no longer needed.
/// @param ptr Pointer to the memory to free
void __moore_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_RUNTIME_MOORERUNTIME_H
