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

/// Pop an element from the back of a queue into a provided buffer.
/// Used for complex types (structs, pointers) that don't fit in int64_t.
/// @param queue Pointer to the queue structure (modified in place)
/// @param result_ptr Pointer to buffer where the element will be written
/// @param element_size Size of the element in bytes
void __moore_queue_pop_back_ptr(MooreQueue *queue, void *result_ptr,
                                int64_t element_size);

/// Pop an element from the front of a queue.
/// @param queue Pointer to the queue structure (modified in place)
/// @param element_size Size of the element in bytes
/// @return The popped element value (as 64-bit integer)
int64_t __moore_queue_pop_front(MooreQueue *queue, int64_t element_size);

/// Pop an element from the front of a queue into a provided buffer.
/// Used for complex types (structs, pointers) that don't fit in int64_t.
/// @param queue Pointer to the queue structure (modified in place)
/// @param result_ptr Pointer to buffer where the element will be written
/// @param element_size Size of the element in bytes
void __moore_queue_pop_front_ptr(MooreQueue *queue, void *result_ptr,
                                 int64_t element_size);

/// Clear all elements from a queue.
/// This corresponds to the SystemVerilog `.delete()` method without arguments.
/// @param queue Pointer to the queue structure (modified in place)
void __moore_queue_clear(MooreQueue *queue);

/// Delete an element at a specific index from a queue.
/// This corresponds to the SystemVerilog `.delete(index)` method.
/// Elements after the deleted index are shifted down.
/// @param queue Pointer to the queue structure (modified in place)
/// @param index Index of the element to delete
/// @param element_size Size of each element in bytes
void __moore_queue_delete_index(MooreQueue *queue, int32_t index,
                                int64_t element_size);

/// Insert an element at a specific index in a queue.
/// This corresponds to the SystemVerilog `.insert(index, item)` method.
/// Elements at and after the index are shifted up by one position.
/// If index < 0, it's treated as 0. If index >= size, the item is appended.
/// @param queue Pointer to the queue structure (modified in place)
/// @param index Index at which to insert the element
/// @param element Pointer to the element to insert
/// @param element_size Size of the element in bytes
void __moore_queue_insert(MooreQueue *queue, int32_t index, void *element,
                          int64_t element_size);

/// Sort a queue and return a new sorted queue.
/// @param queue Pointer to the queue structure
/// @param elem_size Size of each element in bytes
/// @param compare Comparison function (same signature as qsort compare)
/// @return A new queue with sorted elements
void *__moore_queue_sort(void *queue, int64_t elem_size,
                         int (*compare)(const void *, const void *));

/// Reverse sort a queue in place (descending order).
/// @param queue Pointer to the queue structure (modified in place)
/// @param elem_size Size of each element in bytes
void __moore_queue_rsort(MooreQueue *queue, int64_t elem_size);

/// Shuffle a queue in place.
/// @param queue Pointer to the queue structure (modified in place)
/// @param elem_size Size of each element in bytes
void __moore_queue_shuffle(MooreQueue *queue, int64_t elem_size);

/// Reverse a queue in place.
/// @param queue Pointer to the queue structure (modified in place)
/// @param elem_size Size of each element in bytes
void __moore_queue_reverse(MooreQueue *queue, int64_t elem_size);

/// Slice a queue with an inclusive range [start, end].
/// @param queue Pointer to the queue structure
/// @param start Start index (inclusive)
/// @param end End index (inclusive)
/// @param element_size Size of each element in bytes
/// @return A new queue containing the selected slice
MooreQueue __moore_queue_slice(MooreQueue *queue, int64_t start, int64_t end,
                               int64_t element_size);

/// Concatenate multiple queues into a single queue.
/// @param queues Pointer to an array of queues
/// @param count Number of queues in the array
/// @param element_size Size of each element in bytes
/// @return A new queue containing all elements in order
MooreQueue __moore_queue_concat(MooreQueue *queues, int64_t count,
                                int64_t element_size);

/// Get the size (number of elements) of a queue.
/// Implements SystemVerilog queue.size() method.
/// @param queue Pointer to the queue structure
/// @return Number of elements in the queue
int64_t __moore_queue_size(MooreQueue *queue);

/// Get unique elements from a queue.
/// Implements SystemVerilog queue.unique() method.
/// This is a simplified version that doesn't take element_size - it's
/// computed from the stored queue metadata or uses a default.
/// @param queue Pointer to the queue structure
/// @return A new queue containing unique elements (first occurrence of each)
MooreQueue __moore_queue_unique(MooreQueue *queue);

/// Sort a queue in place (ascending order).
/// Implements SystemVerilog queue.sort() method for in-place sorting.
/// @param queue Pointer to the queue structure (modified in place)
/// @param elem_size Size of each element in bytes
void __moore_queue_sort_inplace(MooreQueue *queue, int64_t elem_size);

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

/// Type tag for associative arrays to distinguish string vs integer keys.
typedef enum {
  AssocArrayType_StringKey = 0,
  AssocArrayType_IntKey = 1
} AssocArrayType;

/// Header structure for associative arrays.
/// The interpreter uses this to determine key type for proper marshalling.
typedef struct {
  AssocArrayType type;
  void *array;
} AssocArrayHeader;

/// Create a new empty associative array.
/// @param key_size Size of keys in bytes (0 for string keys)
/// @param value_size Size of values in bytes
/// @return Pointer to the new associative array (opaque handle)
void *__moore_assoc_create(int32_t key_size, int32_t value_size);

/// Get the number of elements in an associative array.
/// @param array Pointer to the associative array
/// @return Number of key-value pairs in the array
int64_t __moore_assoc_size(void *array);

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

/// Check if a key exists in the associative array.
/// @param array Pointer to the associative array
/// @param key Pointer to the key to check
/// @return 1 if key exists, 0 otherwise
int32_t __moore_assoc_exists(void *array, void *key);

/// Get or create a reference to an element in the associative array.
/// If the key doesn't exist, creates a new entry with zero-initialized value.
/// @param array Pointer to the associative array
/// @param key Pointer to the key
/// @param value_size Size of the value in bytes
/// @return Pointer to the value storage
void *__moore_assoc_get_ref(void *array, void *key, int32_t value_size);

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

/// Convert a packed string (integer with ASCII bytes) to a runtime string.
/// SystemVerilog string literals are packed as integers with characters in
/// big-endian order (first char in MSB). This function unpacks them.
/// @param value The packed string value (up to 8 characters in 64-bit)
/// @return A new string with the unpacked characters
MooreString __moore_packed_string_to_string(int64_t value);

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

/// Unpack bits from a value into a dynamic array.
/// The inverse of __moore_stream_concat_bits. Used for streaming unpacking
/// assignments like: {<<{array}} = source_bits
/// @param array Pointer to the destination queue/array
/// @param sourceBits The bits to unpack (up to 64 bits)
/// @param elementBitWidth Bit width of each element
/// @param isRightToLeft If true, unpack right-to-left; otherwise left-to-right
void __moore_stream_unpack_bits(MooreQueue *array, int64_t sourceBits,
                                int32_t elementBitWidth, bool isRightToLeft);

//===----------------------------------------------------------------------===//
// Event Operations
//===----------------------------------------------------------------------===//

/// Trigger an event.
/// Implements the SystemVerilog `->event` syntax for triggering an event.
/// This sets the event flag to true for the current time slot.
/// @param event Pointer to the event (stored as a boolean flag)
void __moore_event_trigger(bool *event);

/// Check if an event was triggered in the current time slot.
/// Implements the SystemVerilog `.triggered` property on events.
/// @param event Pointer to the event (stored as a boolean flag)
/// @return true if the event was triggered, false otherwise
bool __moore_event_triggered(bool *event);

//===----------------------------------------------------------------------===//
// Process Control Operations
//===----------------------------------------------------------------------===//
//
// The SystemVerilog `process` class provides process control capabilities.
// IEEE 1800-2017 Section 9.7 "Process control"
//

/// Get a handle to the currently executing process.
/// Implements SystemVerilog `process::self()` static method.
/// Returns a non-null handle when called from within a process context
/// (llhd.process, initial block, always block, fork branch), or null
/// when called from outside a process context.
/// @return A process handle (non-null if inside a process, null otherwise)
void *__moore_process_self(void);

//===----------------------------------------------------------------------===//
// Mailbox Operations (Inter-process Communication)
//===----------------------------------------------------------------------===//
//
// Mailboxes provide a FIFO-based message passing mechanism for inter-process
// communication in SystemVerilog. Messages are stored as opaque 64-bit values
// (typically handles to actual data structures).
//
// These are stub declarations - the actual implementation is in the
// SyncPrimitivesManager class, accessed via DPI hooks in the interpreter.
//

/// Create a new mailbox.
/// Implements SystemVerilog `mailbox#(T) mbox = new(bound)`.
/// @param bound Maximum number of messages (0 = unbounded)
/// @return Unique mailbox identifier (0 = invalid)
int64_t __moore_mailbox_create(int32_t bound);

/// Try to put a message into a mailbox (non-blocking).
/// Implements SystemVerilog `mailbox.try_put(msg)`.
/// @param mbox_id Mailbox identifier from __moore_mailbox_create
/// @param msg The message to put (as 64-bit value/handle)
/// @return true if the message was put successfully, false if mailbox is full
bool __moore_mailbox_tryput(int64_t mbox_id, int64_t msg);

/// Try to get a message from a mailbox (non-blocking).
/// Implements SystemVerilog `mailbox.try_get(msg)`.
/// @param mbox_id Mailbox identifier from __moore_mailbox_create
/// @param msg_out Pointer to store the retrieved message
/// @return true if a message was retrieved, false if mailbox is empty
bool __moore_mailbox_tryget(int64_t mbox_id, int64_t *msg_out);

/// Get the number of messages in a mailbox.
/// Implements SystemVerilog `mailbox.num()`.
/// @param mbox_id Mailbox identifier from __moore_mailbox_create
/// @return Number of messages currently in the mailbox
int64_t __moore_mailbox_num(int64_t mbox_id);

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
// Randomization Operations
//===----------------------------------------------------------------------===//

/// Basic randomization - fills class memory with random values.
/// This is a simplified implementation of SystemVerilog's randomize() method
/// that fills the entire object with random bytes. A more sophisticated
/// implementation would use class metadata to identify which fields are marked
/// with `rand` and only randomize those.
///
/// @param classPtr Pointer to the class instance to randomize
/// @param classSize Size of the class in bytes
/// @return 1 on success, 0 on failure (e.g., null pointer)
int32_t __moore_randomize_basic(void *classPtr, int64_t classSize);

/// Generate the next randc value for a field.
/// Uses a per-field cycle for small bit widths; larger widths fall back to
/// random values.
/// @param fieldPtr Pointer to the field storage (used as cycle key)
/// @param bitWidth Width of the field in bits
/// @return Next value in the cycle or a random value
int64_t __moore_randc_next(void *fieldPtr, int64_t bitWidth);

/// Generate a weighted random value from a distribution.
/// Implements SystemVerilog distribution constraints (dist keyword).
///
/// The ranges array contains pairs of [low, high] values defining the ranges.
/// The weights array specifies the weight for each range.
/// The perRange array indicates the weight type:
///   - 0 (:=) means the weight applies to each value in the range
///   - 1 (:/) means the weight is divided among values in the range
///
/// Example: x dist { 0 := 10, [1:5] :/ 50, 6 := 40 }
///   ranges: [0, 0, 1, 5, 6, 6] (pairs)
///   weights: [10, 50, 40]
///   perRange: [0, 1, 0]
///
/// @param ranges Array of range pairs [low1, high1, low2, high2, ...]
/// @param weights Array of weights for each range
/// @param perRange Array indicating weight type (0 = :=, 1 = :/)
/// @param numRanges Number of ranges (weights and perRange have this length)
/// @return A random value selected according to the distribution
int64_t __moore_randomize_with_dist(int64_t *ranges, int64_t *weights,
                                    int64_t *perRange, int64_t numRanges);

//===----------------------------------------------------------------------===//
// Dynamic Cast / RTTI Operations
//===----------------------------------------------------------------------===//

/// Perform a dynamic cast check for class hierarchy type checking.
/// Implements the SystemVerilog $cast system function's runtime check.
///
/// The check determines if a source object can be safely downcast to a target
/// type by comparing the object's runtime type ID against the target type ID.
/// A cast succeeds if the object's actual type is the same as, or derived
/// from, the target type.
///
/// IEEE 1800-2017 Section 8.16: "$cast shall return 1 if the cast is legal
/// at runtime, 0 otherwise."
///
/// @param srcTypeId The runtime type ID of the source object (from vtable)
/// @param targetTypeId The type ID of the target (destination) type
/// @param inheritanceDepth Depth of inheritance chain for the target type
/// @return true if cast is valid (src is same or derived from target), false otherwise
bool __moore_dyn_cast_check(int32_t srcTypeId, int32_t targetTypeId,
                            int32_t inheritanceDepth);

//===----------------------------------------------------------------------===//
// Array Locator Methods
//===----------------------------------------------------------------------===//

/// Predicate function type for array locator methods.
/// @param element Pointer to the current element being tested
/// @param userData User-provided context data
/// @return true if the element matches the predicate, false otherwise
typedef bool (*MooreLocatorPredicate)(void *element, void *userData);

/// Find elements in an array that match a predicate.
/// Implements SystemVerilog array locator methods: find, find_first, find_last,
/// find_index, find_first_index, find_last_index.
/// @param array Pointer to the input array/queue
/// @param elementSize Size of each element in bytes
/// @param predicate Function pointer for the predicate (returns bool for each element)
/// @param userData User data to pass to the callback
/// @param mode 0=all, 1=first, 2=last
/// @param returnIndices If true, return indices instead of elements
/// @return A new queue with matching elements or indices
MooreQueue __moore_array_locator(MooreQueue *array, int64_t elementSize,
                                 MooreLocatorPredicate predicate,
                                 void *userData, int32_t mode,
                                 bool returnIndices);

/// Find elements equal to a given value (simpler case without callback).
/// Useful when the comparison can be done by simple memory comparison.
/// @param array Pointer to the input array/queue
/// @param elementSize Size of each element in bytes
/// @param value Pointer to the value to search for
/// @param mode 0=all, 1=first, 2=last
/// @param returnIndices If true, return indices instead of elements
/// @return A new queue with matching elements or indices
MooreQueue __moore_array_find_eq(MooreQueue *array, int64_t elementSize,
                                 void *value, int32_t mode, bool returnIndices);

/// Comparison mode for __moore_array_find_cmp.
/// These values match the order expected by the MooreToCore lowering.
enum MooreCmpMode {
  MOORE_CMP_EQ = 0,  ///< Equal (==)
  MOORE_CMP_NE = 1,  ///< Not equal (!=)
  MOORE_CMP_SGT = 2, ///< Signed greater than (>)
  MOORE_CMP_SGE = 3, ///< Signed greater than or equal (>=)
  MOORE_CMP_SLT = 4, ///< Signed less than (<)
  MOORE_CMP_SLE = 5  ///< Signed less than or equal (<=)
};

/// Find elements matching a comparison predicate against a given value.
/// This is a generalized version of __moore_array_find_eq that supports
/// multiple comparison predicates for array locator methods.
/// @param array Pointer to the input array/queue
/// @param elementSize Size of each element in bytes
/// @param value Pointer to the value to compare against
/// @param cmpMode Comparison mode (0=eq, 1=ne, 2=sgt, 3=sge, 4=slt, 5=sle)
/// @param locatorMode 0=all, 1=first, 2=last
/// @param returnIndices If true, return indices instead of elements
/// @return A new queue with matching elements or indices
MooreQueue __moore_array_find_cmp(MooreQueue *array, int64_t elementSize,
                                  void *value, int32_t cmpMode,
                                  int32_t locatorMode, bool returnIndices);

/// Find elements by comparing a field within each element to a given value.
/// Used for predicates like `arr.find(item) with (item.field == val)`.
/// @param array Pointer to the input array/queue
/// @param elementSize Size of each element in bytes
/// @param fieldOffset Byte offset of the field within each element
/// @param fieldSize Size of the field in bytes
/// @param value Pointer to the value to compare against
/// @param cmpMode Comparison mode (0=eq, 1=ne, 2=sgt, 3=sge, 4=slt, 5=sle)
/// @param locatorMode 0=all, 1=first, 2=last
/// @param returnIndices If true, return indices instead of elements
/// @return A new queue with matching elements or indices
MooreQueue __moore_array_find_field_cmp(MooreQueue *array, int64_t elementSize,
                                        int64_t fieldOffset, int64_t fieldSize,
                                        void *value, int32_t cmpMode,
                                        int32_t locatorMode, bool returnIndices);

/// Find the minimum element(s) in an array.
/// Implements SystemVerilog min() array locator method.
/// @param array Pointer to the input array/queue
/// @param elementSize Size of each element in bytes
/// @param isSigned If true, compare as signed integers; otherwise unsigned
/// @return A new queue containing the minimum element(s)
MooreQueue __moore_array_min(MooreQueue *array, int64_t elementSize,
                             bool isSigned);

/// Find the maximum element(s) in an array.
/// Implements SystemVerilog max() array locator method.
/// @param array Pointer to the input array/queue
/// @param elementSize Size of each element in bytes
/// @param isSigned If true, compare as signed integers; otherwise unsigned
/// @return A new queue containing the maximum element(s)
MooreQueue __moore_array_max(MooreQueue *array, int64_t elementSize,
                             bool isSigned);

/// Find unique elements in an array.
/// Implements SystemVerilog unique() array locator method.
/// @param array Pointer to the input array/queue
/// @param elementSize Size of each element in bytes
/// @return A new queue containing unique elements (first occurrence of each)
MooreQueue __moore_array_unique(MooreQueue *array, int64_t elementSize);

/// Find indices of unique elements in an array.
/// Implements SystemVerilog unique_index() array locator method.
/// @param array Pointer to the input array/queue
/// @param elementSize Size of each element in bytes
/// @return A new queue containing indices of unique elements
MooreQueue __moore_array_unique_index(MooreQueue *array, int64_t elementSize);

/// Reduce array elements by summation.
/// @param array Pointer to the input array/queue
/// @param elementSize Size of each element in bytes
/// @return The sum of all elements (truncated to 64 bits)
int64_t __moore_array_reduce_sum(MooreQueue *array, int64_t elementSize);

/// Reduce array elements by product.
/// @param array Pointer to the input array/queue
/// @param elementSize Size of each element in bytes
/// @return The product of all elements (truncated to 64 bits)
int64_t __moore_array_reduce_product(MooreQueue *array, int64_t elementSize);

/// Reduce array elements by bitwise AND.
/// @param array Pointer to the input array/queue
/// @param elementSize Size of each element in bytes
/// @return Bitwise AND of all elements (truncated to 64 bits)
int64_t __moore_array_reduce_and(MooreQueue *array, int64_t elementSize);

/// Reduce array elements by bitwise OR.
/// @param array Pointer to the input array/queue
/// @param elementSize Size of each element in bytes
/// @return Bitwise OR of all elements (truncated to 64 bits)
int64_t __moore_array_reduce_or(MooreQueue *array, int64_t elementSize);

/// Reduce array elements by bitwise XOR.
/// @param array Pointer to the input array/queue
/// @param elementSize Size of each element in bytes
/// @return Bitwise XOR of all elements (truncated to 64 bits)
int64_t __moore_array_reduce_xor(MooreQueue *array, int64_t elementSize);

//===----------------------------------------------------------------------===//
// Coverage Collection Operations
//===----------------------------------------------------------------------===//
//
// These functions provide runtime support for SystemVerilog coverage
// collection. Covergroups track which values have been observed during
// simulation, enabling functional coverage analysis.
//
// Note: This is basic coverage collection. Full SystemVerilog coverage
// semantics (bins, crosses, options) is future work.
//

/// Coverpoint data structure for tracking sampled values.
/// @member name Name of the coverpoint
/// @member bins Array of bin hit counts (NULL for auto bins)
/// @member num_bins Number of explicit bins (0 for auto bins)
/// @member hits Total number of samples
/// @member min_val Minimum sampled value (for auto bins)
/// @member max_val Maximum sampled value (for auto bins)
typedef struct {
  const char *name;
  int64_t *bins;
  int32_t num_bins;
  int64_t hits;
  int64_t min_val;
  int64_t max_val;
} MooreCoverpoint;

/// Covergroup data structure for grouping coverpoints.
/// @member name Name of the covergroup
/// @member coverpoints Array of coverpoint pointers
/// @member num_coverpoints Number of coverpoints in this group
typedef struct {
  const char *name;
  MooreCoverpoint **coverpoints;
  int32_t num_coverpoints;
} MooreCovergroup;

/// Create a new covergroup instance.
/// Allocates and initializes a covergroup with the specified number of
/// coverpoints. The coverpoints array is allocated but not initialized;
/// use __moore_coverpoint_init to set up each coverpoint.
///
/// @param name Name of the covergroup (for reporting)
/// @param num_coverpoints Number of coverpoints to allocate
/// @return Pointer to the new covergroup, or NULL on allocation failure
void *__moore_covergroup_create(const char *name, int32_t num_coverpoints);

/// Initialize a coverpoint within a covergroup.
/// Sets up a coverpoint with the given name at the specified index.
/// The coverpoint uses automatic bins (tracking min/max values seen).
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint to initialize
/// @param name Name of the coverpoint (for reporting)
void __moore_coverpoint_init(void *cg, int32_t cp_index, const char *name);

/// Destroy a covergroup and free all associated memory.
/// Frees the covergroup, all its coverpoints, and their bin arrays.
///
/// @param cg Pointer to the covergroup to destroy
void __moore_covergroup_destroy(void *cg);

/// Sample a value for a coverpoint.
/// Records that the specified value was observed at the given coverpoint.
/// This increments the hit count and updates min/max tracking.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint within the covergroup
/// @param value The value that was sampled
void __moore_coverpoint_sample(void *cg, int32_t cp_index, int64_t value);

/// Get the coverage percentage for a coverpoint.
/// For auto bins, returns the percentage of the value range that was hit.
/// For explicit bins, returns the percentage of bins that were hit.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @return Coverage percentage (0.0 to 100.0)
double __moore_coverpoint_get_coverage(void *cg, int32_t cp_index);

/// Get the overall coverage percentage for a covergroup.
/// Returns the average coverage across all coverpoints.
/// When per_instance is false (default), this aggregates coverage across
/// all instances of the same covergroup type (by name).
///
/// @param cg Pointer to the covergroup
/// @return Coverage percentage (0.0 to 100.0)
double __moore_covergroup_get_coverage(void *cg);

/// Get the instance-specific coverage percentage for a covergroup.
/// Always returns the coverage for this specific instance, regardless
/// of the per_instance option setting.
/// IEEE 1800-2017 Section 19.8.1: get_inst_coverage()
///
/// @param cg Pointer to the covergroup
/// @return Coverage percentage (0.0 to 100.0)
double __moore_covergroup_get_inst_coverage(void *cg);

/// Get the instance-specific coverage percentage for a coverpoint.
/// For coverpoints, this is equivalent to get_coverage since coverpoints
/// are always instance-specific within their covergroup.
/// IEEE 1800-2017 Section 19.8.1: get_inst_coverage()
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @return Coverage percentage (0.0 to 100.0)
double __moore_coverpoint_get_inst_coverage(void *cg, int32_t cp_index);

/// Print a coverage report for all registered covergroups.
/// Outputs coverage statistics to stdout, including:
/// - Each covergroup name and overall coverage
/// - Each coverpoint name, hit count, and coverage percentage
///
/// This function should be called at the end of simulation (e.g., in $finish).
void __moore_coverage_report(void);

/// Coverage bin types for explicit bin definitions.
/// These types correspond to SystemVerilog bin categories.
enum MooreBinType {
  MOORE_BIN_VALUE = 0,     ///< Single value bin: bins x = {5};
  MOORE_BIN_RANGE = 1,     ///< Range bin: bins x = {[0:15]};
  MOORE_BIN_WILDCARD = 2,  ///< Wildcard bin: bins x = {4'b1???};
  MOORE_BIN_TRANSITION = 3 ///< Transition bin: bins x = (1 => 2);
};

/// Coverage bin kind (normal, illegal, or ignore).
/// These correspond to SystemVerilog bin declarations.
/// IEEE 1800-2017 Section 19.5.
enum MooreBinKind {
  MOORE_BIN_KIND_NORMAL = 0,  ///< Regular bin: bins x = {...}
  MOORE_BIN_KIND_ILLEGAL = 1, ///< Illegal bin: illegal_bins x = {...}
  MOORE_BIN_KIND_IGNORE = 2   ///< Ignore bin: ignore_bins x = {...}
};

/// Transition repeat kinds for sequence coverage bins.
/// Corresponds to SystemVerilog transition repeat specifications.
enum MooreTransitionRepeatKind {
  MOORE_TRANS_NONE = 0,           ///< No repeat: (a => b)
  MOORE_TRANS_CONSECUTIVE = 1,    ///< Consecutive repeat [*n]: (a [*3] => b)
  MOORE_TRANS_NONCONSECUTIVE = 2, ///< Non-consecutive repeat [=n]: (a [=3] => b)
  MOORE_TRANS_GOTO = 3            ///< Goto repeat [->n]: (a [->3] => b)
};

/// Transition step structure for defining a single step in a transition sequence.
/// @member value The value at this step
/// @member repeat_kind Type of repeat (none, consecutive, non-consecutive, goto)
/// @member repeat_from Minimum repeat count (0 if no repeat)
/// @member repeat_to Maximum repeat count (0 if no repeat, equals from for exact)
typedef struct {
  int64_t value;
  int32_t repeat_kind;
  int32_t repeat_from;
  int32_t repeat_to;
} MooreTransitionStep;

/// Transition sequence structure for transition coverage bins.
/// @member steps Array of transition steps
/// @member num_steps Number of steps in the sequence
typedef struct {
  MooreTransitionStep *steps;
  int32_t num_steps;
} MooreTransitionSequence;

/// Coverage bin definition structure.
/// @member name Name of the bin
/// @member type Type of bin (value, range, wildcard, transition)
/// @member kind Kind of bin (normal, illegal, ignore)
/// @member low For value/range bins: lower bound of range (or single value).
///             For wildcard bins: the pattern value (don't care bits are 0).
/// @member high For value/range bins: upper bound of range (same as low for value bins).
///              For wildcard bins: the mask (1 = don't care, 0 = must match).
///              Wildcard match formula: ((value ^ low) & ~high) == 0
/// @member hit_count Number of times this bin was hit
typedef struct {
  const char *name;
  int32_t type;
  int32_t kind;  ///< MooreBinKind: normal, illegal, or ignore
  int64_t low;
  int64_t high;
  int64_t hit_count;
} MooreCoverageBin;

/// Initialize a coverpoint with explicit bin definitions.
/// Sets up a coverpoint with named bins for precise coverage tracking.
/// This provides finer-grained coverage than auto bins.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint to initialize
/// @param name Name of the coverpoint (for reporting)
/// @param bins Array of bin definitions
/// @param num_bins Number of bins in the array
void __moore_coverpoint_init_with_bins(void *cg, int32_t cp_index,
                                       const char *name,
                                       MooreCoverageBin *bins,
                                       int32_t num_bins);

/// Add a single bin to an existing coverpoint.
/// Can be used to dynamically add bins after initialization.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param bin_name Name of the bin
/// @param bin_type Type of bin (see MooreBinType)
/// @param low Lower bound of the bin range
/// @param high Upper bound of the bin range
void __moore_coverpoint_add_bin(void *cg, int32_t cp_index,
                                const char *bin_name, int32_t bin_type,
                                int64_t low, int64_t high);

/// Get the hit count for a specific bin.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param bin_index Index of the bin
/// @return Number of times the bin was hit
int64_t __moore_coverpoint_get_bin_hits(void *cg, int32_t cp_index,
                                        int32_t bin_index);

//===----------------------------------------------------------------------===//
// Transition Coverage Operations
//===----------------------------------------------------------------------===//
//
// Transition coverage tracks state machine transitions rather than just values.
// This implements SystemVerilog transition bins like: bins x = (IDLE => RUN);
//

/// Opaque handle to a transition tracker state machine.
typedef struct MooreTransitionTracker *MooreTransitionTrackerHandle;

/// Create a new transition tracker for a coverpoint.
/// The tracker maintains state to detect multi-step transitions.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @return Handle to the tracker, or NULL on failure
MooreTransitionTrackerHandle __moore_transition_tracker_create(void *cg,
                                                                int32_t cp_index);

/// Destroy a transition tracker and free resources.
///
/// @param tracker Handle to the transition tracker
void __moore_transition_tracker_destroy(MooreTransitionTrackerHandle tracker);

/// Add a transition bin to a coverpoint.
/// A transition bin tracks a sequence of state transitions.
///
/// Example: bins idle_to_run = (IDLE => RUN);
///   sequences = {{IDLE, 0, 0, 0}, {RUN, 0, 0, 0}}
///   num_sequences = 1, each sequence has num_steps = 2
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param bin_name Name of the bin
/// @param sequences Array of transition sequences (alternatives)
/// @param num_sequences Number of alternative sequences
void __moore_coverpoint_add_transition_bin(void *cg, int32_t cp_index,
                                           const char *bin_name,
                                           MooreTransitionSequence *sequences,
                                           int32_t num_sequences);

/// Update transition tracker with a new sampled value.
/// This function advances the state machine for all transition bins
/// and records hits when complete sequences are observed.
///
/// @param tracker Handle to the transition tracker
/// @param value The newly sampled value
void __moore_transition_tracker_sample(MooreTransitionTrackerHandle tracker,
                                       int64_t value);

/// Reset all transition tracker state machines to initial state.
/// Useful when restarting coverage collection or on reset events.
///
/// @param tracker Handle to the transition tracker
void __moore_transition_tracker_reset(MooreTransitionTrackerHandle tracker);

/// Get the number of times a transition bin was hit.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param bin_index Index of the transition bin
/// @return Number of complete transitions observed
int64_t __moore_transition_bin_get_hits(void *cg, int32_t cp_index,
                                        int32_t bin_index);

/// Write a JSON coverage report to a file.
/// Outputs coverage data in JSON format suitable for post-processing.
/// The JSON includes covergroups, coverpoints, bins, and all hit data.
///
/// @param filename Path to the output file (null-terminated string)
/// @return 0 on success, non-zero on failure
int32_t __moore_coverage_report_json(const char *filename);

/// Write a JSON coverage report string to stdout.
/// Useful for debugging and piping to other tools.
void __moore_coverage_report_json_stdout(void);

/// Get coverage data as a JSON string.
/// Allocates and returns a JSON string with all coverage data.
/// Caller is responsible for freeing the returned string with __moore_free.
///
/// @return Allocated JSON string, or NULL on failure
char *__moore_coverage_get_json(void);

//===----------------------------------------------------------------------===//
// Cross Coverage Operations
//===----------------------------------------------------------------------===//
//
// Cross coverage tracks combinations of values from multiple coverpoints.
// This implements the SystemVerilog `cross` construct within covergroups.
//

/// Cross coverage structure for tracking value combinations.
/// @member name Name of the cross
/// @member cp_indices Array of coverpoint indices participating in the cross
/// @member num_cps Number of coverpoints in the cross
/// @member bins Map of value combinations to hit counts (internal)
typedef struct {
  const char *name;
  int32_t *cp_indices;
  int32_t num_cps;
  void *bins_data; // Opaque pointer to internal cross bin data
} MooreCrossCoverage;

/// Add a cross coverage item to a covergroup.
/// Creates a cross that tracks combinations of the specified coverpoints.
///
/// @param cg Pointer to the covergroup
/// @param name Name of the cross (for reporting)
/// @param cp_indices Array of coverpoint indices to cross
/// @param num_cps Number of coverpoints in the cross (typically 2 or more)
/// @return Index of the created cross, or -1 on failure
int32_t __moore_cross_create(void *cg, const char *name, int32_t *cp_indices,
                             int32_t num_cps);

/// Sample all crosses in a covergroup.
/// Should be called after sampling all coverpoints to update cross bins.
///
/// @param cg Pointer to the covergroup
/// @param cp_values Array of sampled values for each coverpoint
/// @param num_values Number of values (must match num_coverpoints)
void __moore_cross_sample(void *cg, int64_t *cp_values, int32_t num_values);

/// Get the coverage percentage for a specific cross.
/// Respects the at_least threshold from covergroup options.
///
/// @param cg Pointer to the covergroup
/// @param cross_index Index of the cross
/// @return Coverage percentage (0.0 to 100.0)
double __moore_cross_get_coverage(void *cg, int32_t cross_index);

/// Get the instance-specific coverage percentage for a cross.
/// For crosses, this is equivalent to get_coverage since crosses
/// are always instance-specific within their covergroup.
/// IEEE 1800-2017 Section 19.8.1: get_inst_coverage()
///
/// @param cg Pointer to the covergroup
/// @param cross_index Index of the cross
/// @return Coverage percentage (0.0 to 100.0)
double __moore_cross_get_inst_coverage(void *cg, int32_t cross_index);

/// Get the total number of cross bins hit.
///
/// @param cg Pointer to the covergroup
/// @param cross_index Index of the cross
/// @return Number of unique cross bin combinations that were hit
int64_t __moore_cross_get_bins_hit(void *cg, int32_t cross_index);

//===----------------------------------------------------------------------===//
// Cross Coverage Named Bins and Filtering
//===----------------------------------------------------------------------===//
//
// Enhanced cross coverage supporting named bins with binsof expressions,
// ignore_bins, and illegal_bins. This implements SystemVerilog cross coverage
// constructs like:
//   cross cp1, cp2 {
//     bins both_low = binsof(cp1.low) && binsof(cp2.low);
//     ignore_bins skip = binsof(cp1.high) && binsof(cp2.low);
//     illegal_bins bad = binsof(cp1) intersect {0} && binsof(cp2) intersect {0};
//   }
//

/// Cross bin filter type for named cross bins.
/// Specifies how a named cross bin filters the cross product space.
enum MooreCrossBinKind {
  MOORE_CROSS_BIN_NORMAL = 0,  ///< Normal named cross bin
  MOORE_CROSS_BIN_IGNORE = 1,  ///< Ignore bin (excluded from coverage)
  MOORE_CROSS_BIN_ILLEGAL = 2  ///< Illegal bin (triggers error if hit)
};

/// Binsof filter for a single coverpoint in a cross bin expression.
/// Specifies which bins of a coverpoint are included in the cross bin.
/// @member cp_index Index of the coverpoint in the cross
/// @member bin_indices Array of bin indices to include (NULL = all bins)
/// @member num_bins Number of bin indices (0 = all bins)
/// @member values Array of specific values to intersect (NULL = no value filter)
/// @member num_values Number of intersect values (0 = no value filter)
/// @member negate If true, negate the filter (!binsof)
typedef struct {
  int32_t cp_index;
  int32_t *bin_indices;
  int32_t num_bins;
  int64_t *values;
  int32_t num_values;
  bool negate;
} MooreCrossBinsofFilter;

/// Named cross bin definition.
/// @member name Name of the cross bin
/// @member kind Type of bin (normal, ignore, illegal)
/// @member filters Array of binsof filters (AND-ed together)
/// @member num_filters Number of filters in the expression
/// @member hit_count Number of times this named bin was hit
typedef struct {
  const char *name;
  int32_t kind;
  MooreCrossBinsofFilter *filters;
  int32_t num_filters;
  int64_t hit_count;
} MooreCrossBinDef;

/// Add a named bin to a cross coverage item.
/// Named bins allow filtering the cross product space using binsof expressions.
///
/// Example: bins both_high = binsof(cp1.high) && binsof(cp2.high);
///
/// @param cg Pointer to the covergroup
/// @param cross_index Index of the cross (from __moore_cross_create)
/// @param name Name of the cross bin
/// @param kind Type of bin (normal, ignore, illegal)
/// @param filters Array of binsof filters
/// @param num_filters Number of filters
/// @return Index of the created cross bin, or -1 on failure
int32_t __moore_cross_add_named_bin(void *cg, int32_t cross_index,
                                     const char *name, int32_t kind,
                                     MooreCrossBinsofFilter *filters,
                                     int32_t num_filters);

/// Add an ignore_bins entry to a cross coverage item.
/// Shorthand for __moore_cross_add_named_bin with MOORE_CROSS_BIN_IGNORE.
///
/// @param cg Pointer to the covergroup
/// @param cross_index Index of the cross
/// @param name Name of the ignore bin
/// @param filters Array of binsof filters
/// @param num_filters Number of filters
/// @return Index of the created cross bin, or -1 on failure
int32_t __moore_cross_add_ignore_bin(void *cg, int32_t cross_index,
                                      const char *name,
                                      MooreCrossBinsofFilter *filters,
                                      int32_t num_filters);

/// Add an illegal_bins entry to a cross coverage item.
/// Shorthand for __moore_cross_add_named_bin with MOORE_CROSS_BIN_ILLEGAL.
///
/// @param cg Pointer to the covergroup
/// @param cross_index Index of the cross
/// @param name Name of the illegal bin
/// @param filters Array of binsof filters
/// @param num_filters Number of filters
/// @return Index of the created cross bin, or -1 on failure
int32_t __moore_cross_add_illegal_bin(void *cg, int32_t cross_index,
                                       const char *name,
                                       MooreCrossBinsofFilter *filters,
                                       int32_t num_filters);

/// Get the hit count for a named cross bin.
///
/// @param cg Pointer to the covergroup
/// @param cross_index Index of the cross
/// @param bin_index Index of the named cross bin
/// @return Number of times the named bin was hit
int64_t __moore_cross_get_named_bin_hits(void *cg, int32_t cross_index,
                                          int32_t bin_index);

/// Check if a value tuple matches any illegal cross bin.
///
/// @param cg Pointer to the covergroup
/// @param cross_index Index of the cross
/// @param values Array of values (one per coverpoint in the cross)
/// @return true if the value tuple matches an illegal cross bin
bool __moore_cross_is_illegal(void *cg, int32_t cross_index, int64_t *values);

/// Check if a value tuple matches any ignore cross bin.
///
/// @param cg Pointer to the covergroup
/// @param cross_index Index of the cross
/// @param values Array of values (one per coverpoint in the cross)
/// @return true if the value tuple matches an ignore cross bin
bool __moore_cross_is_ignored(void *cg, int32_t cross_index, int64_t *values);

/// Get the number of named bins defined for a cross.
///
/// @param cg Pointer to the covergroup
/// @param cross_index Index of the cross
/// @return Number of named bins (including ignore and illegal bins)
int32_t __moore_cross_get_num_named_bins(void *cg, int32_t cross_index);

/// Illegal cross bin callback function type.
/// Called when an illegal cross bin is hit during sampling.
typedef void (*MooreIllegalCrossBinCallback)(const char *cg_name,
                                              const char *cross_name,
                                              const char *bin_name,
                                              int64_t *values,
                                              int32_t num_values,
                                              void *userData);

/// Register a callback for illegal cross bin hits.
///
/// @param callback The callback function to register
/// @param userData User data to pass to the callback
void __moore_cross_set_illegal_bin_callback(MooreIllegalCrossBinCallback callback,
                                             void *userData);

//===----------------------------------------------------------------------===//
// Coverage Reset and Aggregation
//===----------------------------------------------------------------------===//

/// Reset all coverage data for a covergroup.
/// Clears all hit counts and value trackers, but preserves the structure.
///
/// @param cg Pointer to the covergroup
void __moore_covergroup_reset(void *cg);

/// Reset coverage data for a specific coverpoint.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
void __moore_coverpoint_reset(void *cg, int32_t cp_index);

/// Get total coverage across all registered covergroups.
/// Returns the weighted average coverage of all covergroups.
///
/// @return Total coverage percentage (0.0 to 100.0)
double __moore_coverage_get_total(void);

/// Get the number of registered covergroups.
///
/// @return Number of covergroups currently registered
int32_t __moore_coverage_get_num_covergroups(void);

/// Set a coverage goal for a covergroup.
/// Used for reporting whether coverage targets have been met.
///
/// @param cg Pointer to the covergroup
/// @param goal Coverage goal percentage (0.0 to 100.0)
void __moore_covergroup_set_goal(void *cg, double goal);

/// Get the coverage goal for a covergroup.
///
/// @param cg Pointer to the covergroup
/// @return Coverage goal percentage (default: 100.0)
double __moore_covergroup_get_goal(void *cg);

/// Check if a covergroup has met its coverage goal.
///
/// @param cg Pointer to the covergroup
/// @return true if coverage >= goal, false otherwise
bool __moore_covergroup_goal_met(void *cg);

//===----------------------------------------------------------------------===//
// Coverage Assertion APIs
//===----------------------------------------------------------------------===//
//
// These functions provide assertion-style coverage checking that can be used
// to enforce coverage goals during simulation. When assertions fail, they
// can invoke a callback or print error messages.
//
// Use cases:
// - End-of-simulation coverage checks
// - Continuous coverage monitoring during simulation
// - Integration with UVM scoreboard reporting
// - CI/CD coverage gate enforcement
//

/// Callback function type for coverage assertion failures.
/// Called when a coverage assertion fails (coverage is below the required goal).
///
/// @param cg_name Name of the covergroup that failed (or NULL for global)
/// @param cp_name Name of the coverpoint that failed (or NULL for covergroup-level)
/// @param actual_coverage The actual coverage percentage achieved
/// @param required_goal The required coverage goal percentage
/// @param userData User-provided context data
typedef void (*MooreCoverageAssertCallback)(const char *cg_name,
                                            const char *cp_name,
                                            double actual_coverage,
                                            double required_goal,
                                            void *userData);

/// Set the callback function for coverage assertion failures.
/// When a coverage assertion fails, this callback will be invoked before
/// returning the failure result.
///
/// @param callback Function to call on assertion failure (NULL to disable)
/// @param userData User data passed to the callback
void __moore_coverage_set_failure_callback(MooreCoverageAssertCallback callback,
                                           void *userData);

/// Assert that overall coverage meets a minimum goal percentage.
/// Checks total coverage across all registered covergroups against the goal.
/// If coverage is below the goal, invokes the failure callback (if set).
///
/// @param min_percentage Minimum required coverage percentage (0.0 to 100.0)
/// @return true if coverage >= min_percentage, false otherwise
bool __moore_coverage_assert_goal(double min_percentage);

/// Assert that a covergroup meets a minimum coverage goal.
/// Uses either the specified min_percentage or the covergroup's configured goal,
/// whichever is higher. Invokes failure callback on assertion failure.
///
/// @param cg Pointer to the covergroup
/// @param min_percentage Minimum required coverage percentage (0.0 to 100.0)
/// @return true if covergroup coverage >= goal, false otherwise
bool __moore_covergroup_assert_goal(void *cg, double min_percentage);

/// Assert that a coverpoint meets a minimum coverage goal.
/// Uses either the specified min_percentage or the coverpoint's configured goal,
/// whichever is higher. Invokes failure callback on assertion failure.
///
/// @param cg Pointer to the covergroup containing the coverpoint
/// @param cp_index Index of the coverpoint within the covergroup
/// @param min_percentage Minimum required coverage percentage (0.0 to 100.0)
/// @return true if coverpoint coverage >= goal, false otherwise
bool __moore_coverpoint_assert_goal(void *cg, int32_t cp_index,
                                    double min_percentage);

/// Check if all defined coverage goals are met.
/// Iterates through all registered covergroups and their coverpoints,
/// checking each against its configured goal. Invokes the failure callback
/// for each goal that is not met.
///
/// @return true if all goals are met, false if any goal is not met
bool __moore_coverage_check_all_goals(void);

/// Get the number of coverage goals that are not met.
/// Useful for summary reporting at end of simulation.
///
/// @return Count of covergroups and coverpoints that have not met their goals
int32_t __moore_coverage_get_unmet_goal_count(void);

/// Register a coverage assertion to be checked at simulation end.
/// Multiple assertions can be registered and will all be checked when
/// __moore_coverage_check_registered_assertions() is called.
///
/// @param cg Pointer to covergroup (NULL for global coverage check)
/// @param cp_index Index of coverpoint (-1 for covergroup-level check)
/// @param min_percentage Minimum required coverage percentage
/// @return Assertion ID (>= 0) on success, -1 on failure
int32_t __moore_coverage_register_assertion(void *cg, int32_t cp_index,
                                            double min_percentage);

/// Check all registered coverage assertions.
/// Typically called at end of simulation. Returns true only if all
/// registered assertions pass. Invokes failure callback for each failure.
///
/// @return true if all assertions pass, false if any assertion fails
bool __moore_coverage_check_registered_assertions(void);

/// Clear all registered coverage assertions.
void __moore_coverage_clear_registered_assertions(void);

/// Set the weight for a covergroup (relative importance in coverage calculation).
/// IEEE 1800-2017 Section 19.7.1: option.weight
///
/// @param cg Pointer to the covergroup
/// @param weight Weight value (default: 1)
void __moore_covergroup_set_weight(void *cg, int64_t weight);

/// Get the weight for a covergroup.
///
/// @param cg Pointer to the covergroup
/// @return Weight value (default: 1)
int64_t __moore_covergroup_get_weight(void *cg);

/// Set per_instance mode for a covergroup.
/// When true, coverage is tracked per-instance rather than merged across instances.
/// IEEE 1800-2017 Section 19.7.1: option.per_instance
///
/// @param cg Pointer to the covergroup
/// @param perInstance true for per-instance coverage, false for type-level
void __moore_covergroup_set_per_instance(void *cg, bool perInstance);

/// Get per_instance mode for a covergroup.
///
/// @param cg Pointer to the covergroup
/// @return true if per-instance mode is enabled
bool __moore_covergroup_get_per_instance(void *cg);

/// Set the at_least threshold for a covergroup.
/// Specifies minimum number of hits for a bin to be considered covered.
/// IEEE 1800-2017 Section 19.7.1: option.at_least
///
/// @param cg Pointer to the covergroup
/// @param atLeast Minimum hit count for coverage (default: 1)
void __moore_covergroup_set_at_least(void *cg, int64_t atLeast);

/// Get the at_least threshold for a covergroup.
///
/// @param cg Pointer to the covergroup
/// @return Minimum hit count for coverage (default: 1)
int64_t __moore_covergroup_get_at_least(void *cg);

/// Set the comment string for a covergroup.
/// IEEE 1800-2017 Section 19.7.1: option.comment
///
/// @param cg Pointer to the covergroup
/// @param comment Comment string (copied internally)
void __moore_covergroup_set_comment(void *cg, const char *comment);

/// Get the comment string for a covergroup.
///
/// @param cg Pointer to the covergroup
/// @return Comment string, or NULL if not set
const char *__moore_covergroup_get_comment(void *cg);

/// Set the weight for a coverpoint.
/// IEEE 1800-2017 Section 19.7.2: option.weight for coverpoints
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param weight Weight value (default: 1)
void __moore_coverpoint_set_weight(void *cg, int32_t cp_index, int64_t weight);

/// Get the weight for a coverpoint.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @return Weight value (default: 1)
int64_t __moore_coverpoint_get_weight(void *cg, int32_t cp_index);

/// Set the goal for a coverpoint.
/// IEEE 1800-2017 Section 19.7.2: option.goal for coverpoints
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param goal Target coverage percentage (0.0 to 100.0)
void __moore_coverpoint_set_goal(void *cg, int32_t cp_index, double goal);

/// Get the goal for a coverpoint.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @return Target coverage percentage (default: 100.0)
double __moore_coverpoint_get_goal(void *cg, int32_t cp_index);

/// Set the at_least threshold for a coverpoint.
/// IEEE 1800-2017 Section 19.7.2: option.at_least for coverpoints
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param atLeast Minimum hit count for coverage (default: 1)
void __moore_coverpoint_set_at_least(void *cg, int32_t cp_index,
                                      int64_t atLeast);

/// Get the at_least threshold for a coverpoint.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @return Minimum hit count for coverage (default: 1)
int64_t __moore_coverpoint_get_at_least(void *cg, int32_t cp_index);

/// Set the comment string for a coverpoint.
/// IEEE 1800-2017 Section 19.7.2: option.comment for coverpoints
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param comment Comment string (copied internally)
void __moore_coverpoint_set_comment(void *cg, int32_t cp_index,
                                     const char *comment);

/// Get the comment string for a coverpoint.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @return Comment string, or NULL if not set
const char *__moore_coverpoint_get_comment(void *cg, int32_t cp_index);

/// Set the auto_bin_max for a covergroup.
/// IEEE 1800-2017 Section 19.7.1: option.auto_bin_max
/// Specifies the maximum number of automatically created bins.
///
/// @param cg Pointer to the covergroup
/// @param maxBins Maximum number of auto bins (default: 64)
void __moore_covergroup_set_auto_bin_max(void *cg, int64_t maxBins);

/// Get the auto_bin_max for a covergroup.
///
/// @param cg Pointer to the covergroup
/// @return Maximum number of auto bins (default: 64)
int64_t __moore_covergroup_get_auto_bin_max(void *cg);

/// Set the auto_bin_max for a coverpoint.
/// IEEE 1800-2017 Section 19.7.2: option.auto_bin_max for coverpoints
/// Specifies the maximum number of automatically created bins.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param maxBins Maximum number of auto bins (default: 64)
void __moore_coverpoint_set_auto_bin_max(void *cg, int32_t cp_index,
                                          int64_t maxBins);

/// Get the auto_bin_max for a coverpoint.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @return Maximum number of auto bins (default: 64)
int64_t __moore_coverpoint_get_auto_bin_max(void *cg, int32_t cp_index);

//===----------------------------------------------------------------------===//
// Generic Coverage Option API
//===----------------------------------------------------------------------===//
//
// These functions provide a flexible string-based API for setting and getting
// coverage options. This mirrors the SystemVerilog syntax:
//   cg.option.goal = 90;
//   cg.cp.option.at_least = 5;
//

/// Coverage option identifiers for the generic API.
enum MooreCoverageOption {
  MOORE_OPTION_GOAL = 0,        ///< Coverage goal percentage (double)
  MOORE_OPTION_WEIGHT = 1,      ///< Weight for coverage calculation (int64)
  MOORE_OPTION_AT_LEAST = 2,    ///< Minimum hit count (int64)
  MOORE_OPTION_AUTO_BIN_MAX = 3 ///< Maximum auto bins (int64)
};

/// Set a covergroup option using the generic API.
/// For integer options (weight, at_least, auto_bin_max), the value is cast.
/// For double options (goal), the value is used directly.
///
/// @param cg Pointer to the covergroup
/// @param option Option identifier (see MooreCoverageOption)
/// @param value Option value (interpreted based on option type)
void __moore_covergroup_set_option(void *cg, int32_t option, double value);

/// Get a covergroup option using the generic API.
///
/// @param cg Pointer to the covergroup
/// @param option Option identifier (see MooreCoverageOption)
/// @return Option value (as double, cast appropriately for integer options)
double __moore_covergroup_get_option(void *cg, int32_t option);

/// Set a coverpoint option using the generic API.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param option Option identifier (see MooreCoverageOption)
/// @param value Option value (interpreted based on option type)
void __moore_coverpoint_set_option(void *cg, int32_t cp_index, int32_t option,
                                    double value);

/// Get a coverpoint option using the generic API.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param option Option identifier (see MooreCoverageOption)
/// @return Option value (as double, cast appropriately for integer options)
double __moore_coverpoint_get_option(void *cg, int32_t cp_index, int32_t option);

/// Get weighted coverage for a covergroup.
/// Calculates coverage considering per-coverpoint and per-cross weights.
/// IEEE 1800-2017 Section 19.8: coverage calculation with weights
///
/// @param cg Pointer to the covergroup
/// @return Weighted coverage percentage (0.0 to 100.0)
double __moore_covergroup_get_weighted_coverage(void *cg);

/// Check if a bin meets the at_least threshold.
/// Used for detailed coverage reporting.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param bin_index Index of the bin
/// @return true if bin hits >= at_least threshold
bool __moore_coverpoint_bin_covered(void *cg, int32_t cp_index,
                                     int32_t bin_index);

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
// The exclusion file format is a simple text-based format:
//   # Comment lines start with #
//   # Empty lines are ignored
//   # Format: covergroup_name.coverpoint_name.bin_name
//   # Wildcards: * matches any sequence of characters
//   cg_name.cp_name.bin_name
//   cg_name.cp_name.*        # Exclude all bins in coverpoint
//   cg_name.*.bin_name       # Exclude bin in all coverpoints
//   *.*.excluded_bin         # Exclude bin in all covergroups/coverpoints
//

/// Exclude a bin from coverage calculation.
/// The excluded bin will not count toward coverage goals, but will still
/// track hits for reporting purposes.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param bin_name Name of the bin to exclude
void __moore_coverpoint_exclude_bin(void *cg, int32_t cp_index,
                                     const char *bin_name);

/// Re-include a previously excluded bin in coverage calculation.
/// Removes the bin from the exclusion list.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param bin_name Name of the bin to re-include
void __moore_coverpoint_include_bin(void *cg, int32_t cp_index,
                                     const char *bin_name);

/// Check if a bin is currently excluded from coverage calculation.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param bin_name Name of the bin to check
/// @return true if the bin is excluded, false otherwise
bool __moore_coverpoint_is_bin_excluded(void *cg, int32_t cp_index,
                                         const char *bin_name);

/// Load exclusions from an exclusion file.
/// Parses the file and applies exclusions to all matching covergroups,
/// coverpoints, and bins. The file should use the simple text format
/// described above.
///
/// @param filename Path to the exclusion file (null-terminated string)
/// @return true on success, false if file could not be opened or parsed
bool __moore_covergroup_set_exclusion_file(const char *filename);

/// Get the current exclusion file path.
///
/// @return The path to the currently loaded exclusion file, or NULL if none
const char *__moore_covergroup_get_exclusion_file(void);

/// Get the number of excluded bins for a coverpoint.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @return Number of bins currently excluded
int32_t __moore_coverpoint_get_excluded_bin_count(void *cg, int32_t cp_index);

/// Clear all exclusions for a coverpoint.
/// Removes all bins from the exclusion list for the specified coverpoint.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
void __moore_coverpoint_clear_exclusions(void *cg, int32_t cp_index);

//===----------------------------------------------------------------------===//
// Illegal Bins and Ignore Bins Runtime Support
//===----------------------------------------------------------------------===//
//
// These functions implement runtime enforcement for illegal_bins and
// ignore_bins as specified in IEEE 1800-2017 Section 19.5.
//
// - illegal_bins: Values that should never occur. Hitting an illegal bin
//   triggers an error/warning at runtime.
// - ignore_bins: Values to exclude from coverage calculation. These values
//   do not count toward coverage metrics.
//

/// Result code for illegal bin detection.
enum MooreIllegalBinResult {
  MOORE_ILLEGAL_BIN_OK = 0,       ///< No illegal bin was hit
  MOORE_ILLEGAL_BIN_HIT = 1,      ///< An illegal bin was hit
  MOORE_ILLEGAL_BIN_WARNING = 2   ///< Warning-only mode (non-fatal)
};

/// Illegal bin callback function type.
/// Called when an illegal bin is hit during sampling.
/// @param cg_name Name of the covergroup
/// @param cp_name Name of the coverpoint
/// @param bin_name Name of the illegal bin that was hit
/// @param value The sampled value that matched the illegal bin
/// @param userData User-provided context data
typedef void (*MooreIllegalBinCallback)(const char *cg_name, const char *cp_name,
                                         const char *bin_name, int64_t value,
                                         void *userData);

/// Set illegal bins for a coverpoint.
/// Marks specified value ranges as illegal. Sampling a value that matches
/// an illegal bin will trigger an error or warning.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param bins Array of bin definitions (must have kind = MOORE_BIN_KIND_ILLEGAL)
/// @param num_bins Number of bins in the array
void __moore_coverpoint_set_illegal_bins(void *cg, int32_t cp_index,
                                          MooreCoverageBin *bins,
                                          int32_t num_bins);

/// Set ignore bins for a coverpoint.
/// Marks specified value ranges as ignored. Sampling a value that matches
/// an ignore bin will not count toward coverage metrics.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param bins Array of bin definitions (must have kind = MOORE_BIN_KIND_IGNORE)
/// @param num_bins Number of bins in the array
void __moore_coverpoint_set_ignore_bins(void *cg, int32_t cp_index,
                                         MooreCoverageBin *bins,
                                         int32_t num_bins);

/// Add a single illegal bin to a coverpoint.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param bin_name Name of the illegal bin
/// @param low Lower bound of the illegal range
/// @param high Upper bound of the illegal range
void __moore_coverpoint_add_illegal_bin(void *cg, int32_t cp_index,
                                         const char *bin_name,
                                         int64_t low, int64_t high);

/// Add a single ignore bin to a coverpoint.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param bin_name Name of the ignore bin
/// @param low Lower bound of the ignore range
/// @param high Upper bound of the ignore range
void __moore_coverpoint_add_ignore_bin(void *cg, int32_t cp_index,
                                        const char *bin_name,
                                        int64_t low, int64_t high);

/// Register a callback for illegal bin hits.
/// The callback will be invoked whenever an illegal bin is hit during sampling.
/// Set to NULL to disable the callback.
///
/// @param callback The callback function to register
/// @param userData User data to pass to the callback
void __moore_coverage_set_illegal_bin_callback(MooreIllegalBinCallback callback,
                                                void *userData);

/// Enable or disable fatal errors on illegal bin hits.
/// When enabled (default), hitting an illegal bin will cause the simulation
/// to terminate with an error. When disabled, only a warning is issued.
///
/// @param fatal true for fatal errors, false for warnings only
void __moore_coverage_set_illegal_bin_fatal(bool fatal);

/// Check if illegal bin hits are configured as fatal.
///
/// @return true if illegal bin hits cause fatal errors
bool __moore_coverage_illegal_bin_is_fatal(void);

/// Get the count of illegal bin hits since the start of simulation.
///
/// @return Total number of illegal bin hits
int64_t __moore_coverage_get_illegal_bin_hits(void);

/// Reset the illegal bin hit counter.
void __moore_coverage_reset_illegal_bin_hits(void);

/// Check if a value matches any ignore bin for a coverpoint.
/// This can be used to filter samples before processing.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param value The value to check
/// @return true if the value matches an ignore bin
bool __moore_coverpoint_is_ignored(void *cg, int32_t cp_index, int64_t value);

/// Check if a value matches any illegal bin for a coverpoint.
///
/// @param cg Pointer to the covergroup
/// @param cp_index Index of the coverpoint
/// @param value The value to check
/// @return true if the value matches an illegal bin
bool __moore_coverpoint_is_illegal(void *cg, int32_t cp_index, int64_t value);

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

/// Sample callback function type.
/// Called before or after sampling a covergroup.
/// @param cg Pointer to the covergroup being sampled
/// @param args Array of sample arguments (NULL if no arguments)
/// @param num_args Number of sample arguments (0 if no arguments)
/// @param userData User-provided context data
typedef void (*MooreSampleCallback)(void *cg, int64_t *args, int32_t num_args,
                                     void *userData);

/// Explicitly sample a covergroup with no arguments.
/// Triggers the sample() method on a covergroup, invoking pre/post callbacks.
/// IEEE 1800-2017 Section 19.8: sample() built-in method.
///
/// @param cg Pointer to the covergroup
void __moore_covergroup_sample(void *cg);

/// Explicitly sample a covergroup with arguments.
/// Triggers sample() with the provided arguments, which are passed to
/// coverpoints based on the configured sample argument mapping.
/// IEEE 1800-2017 Section 19.6: covergroup with arguments.
///
/// Example: covergroup cg with function sample(bit [7:0] data, bit valid);
///          cg.sample(my_data, my_valid);
///
/// @param cg Pointer to the covergroup
/// @param args Array of sample argument values
/// @param num_args Number of arguments in the array
void __moore_covergroup_sample_with_args(void *cg, int64_t *args,
                                          int32_t num_args);

/// Register a pre-sample callback for a covergroup.
/// The callback is invoked before any coverpoints are sampled.
/// IEEE 1800-2017 Section 19.8.1: pre_sample() method.
///
/// @param cg Pointer to the covergroup
/// @param callback The callback function to register (NULL to disable)
/// @param userData User data to pass to the callback
void __moore_covergroup_set_pre_sample_callback(
    void *cg, void (*callback)(void *, int64_t *, int32_t, void *),
    void *userData);

/// Register a post-sample callback for a covergroup.
/// The callback is invoked after all coverpoints are sampled.
/// IEEE 1800-2017 Section 19.8.1: post_sample() method.
///
/// @param cg Pointer to the covergroup
/// @param callback The callback function to register (NULL to disable)
/// @param userData User data to pass to the callback
void __moore_covergroup_set_post_sample_callback(
    void *cg, void (*callback)(void *, int64_t *, int32_t, void *),
    void *userData);

/// Register a global pre-sample callback for all covergroups.
/// This callback is invoked before any covergroup-specific callback.
///
/// @param callback The callback function to register (NULL to disable)
/// @param userData User data to pass to the callback
void __moore_coverage_set_global_pre_sample_callback(
    void (*callback)(void *, int64_t *, int32_t, void *), void *userData);

/// Register a global post-sample callback for all covergroups.
/// This callback is invoked after any covergroup-specific callback.
///
/// @param callback The callback function to register (NULL to disable)
/// @param userData User data to pass to the callback
void __moore_coverage_set_global_post_sample_callback(
    void (*callback)(void *, int64_t *, int32_t, void *), void *userData);

/// Set the sample argument mapping for a covergroup.
/// Maps sample() arguments to coverpoint indices.
/// Mapping array: index = coverpoint index, value = argument index (-1 = skip).
///
/// Example: For covergroup with 3 coverpoints and sample(a, b):
///   mapping = {0, 1, -1}  // cp0 gets arg0, cp1 gets arg1, cp2 is not sampled
///
/// @param cg Pointer to the covergroup
/// @param mapping Array of argument indices for each coverpoint
/// @param num_mappings Number of entries in the mapping array
void __moore_covergroup_set_sample_arg_mapping(void *cg, int32_t *mapping,
                                                int32_t num_mappings);

/// Enable or disable sampling for a covergroup.
/// When disabled, sample() calls are ignored.
///
/// @param cg Pointer to the covergroup
/// @param enabled true to enable sampling, false to disable
void __moore_covergroup_set_sample_enabled(void *cg, bool enabled);

/// Check if sampling is enabled for a covergroup.
///
/// @param cg Pointer to the covergroup
/// @return true if sampling is enabled (default), false if disabled
bool __moore_covergroup_is_sample_enabled(void *cg);

/// Set the sample event for a covergroup.
/// Associates a named event with the covergroup for automatic sampling.
/// IEEE 1800-2017 Section 19.3: covergroup cg @(event);
///
/// @param cg Pointer to the covergroup
/// @param eventName Name of the sampling event (NULL to disable)
void __moore_covergroup_set_sample_event(void *cg, const char *eventName);

/// Get the sample event name for a covergroup.
///
/// @param cg Pointer to the covergroup
/// @return The event name, or NULL if no event is set
const char *__moore_covergroup_get_sample_event(void *cg);

/// Check if a covergroup has a sample event configured.
///
/// @param cg Pointer to the covergroup
/// @return true if a sample event is set
bool __moore_covergroup_has_sample_event(void *cg);

/// Trigger a sample event.
/// If the event name matches the covergroup's configured sample event,
/// the covergroup is sampled.
///
/// @param cg Pointer to the covergroup
/// @param eventName Name of the triggered event (NULL triggers any event)
void __moore_covergroup_trigger_sample_event(void *cg, const char *eventName);

//===----------------------------------------------------------------------===//
// Coverage Exclusion API
//===----------------------------------------------------------------------===//
//
// These functions provide runtime coverage exclusion/filtering capabilities.
// Exclusions can be added programmatically or loaded from exclusion files.
// This implements functionality similar to commercial simulator exclusion
// file formats.
//

/// Add a coverage exclusion pattern.
/// Excludes matching covergroups/coverpoints/bins from coverage calculation.
/// Pattern format: "covergroup_name.coverpoint_name.bin_name"
/// Supports wildcards: "*" matches any sequence, "?" matches single character.
///
/// @param pattern The exclusion pattern (null-terminated string)
/// @return 0 on success, non-zero on failure
int32_t __moore_coverage_add_exclusion(const char *pattern);

/// Remove a coverage exclusion pattern.
///
/// @param pattern The pattern to remove
/// @return 0 if pattern was found and removed, non-zero otherwise
int32_t __moore_coverage_remove_exclusion(const char *pattern);

/// Clear all coverage exclusion patterns.
void __moore_coverage_clear_exclusions(void);

/// Load exclusion patterns from a file.
/// File format: one pattern per line, lines starting with '#' are comments.
///
/// @param filename Path to the exclusion file
/// @return Number of patterns loaded, or -1 on error
int32_t __moore_coverage_load_exclusions(const char *filename);

/// Save current exclusion patterns to a file.
///
/// @param filename Path to the output file
/// @return 0 on success, non-zero on failure
int32_t __moore_coverage_save_exclusions(const char *filename);

/// Check if a specific bin is excluded by the current exclusion patterns.
///
/// @param cg_name Name of the covergroup
/// @param cp_name Name of the coverpoint
/// @param bin_name Name of the bin
/// @return true if the bin is excluded
bool __moore_coverage_is_excluded(const char *cg_name, const char *cp_name,
                                   const char *bin_name);

/// Get the number of active exclusion patterns.
///
/// @return Number of exclusion patterns
int32_t __moore_coverage_get_exclusion_count(void);

//===----------------------------------------------------------------------===//
// HTML Coverage Report
//===----------------------------------------------------------------------===//

/// Generate an HTML coverage report file.
/// Creates a self-contained HTML file with interactive coverage visualization.
///
/// @param filename Path to the output HTML file
/// @return 0 on success, non-zero on failure
int32_t __moore_coverage_report_html(const char *filename);

//===----------------------------------------------------------------------===//
// Text Coverage Report
//===----------------------------------------------------------------------===//
//
// Text-based coverage report functions for CI/automation use.
// These provide simple, parseable output that is faster to generate than HTML.
//

/// Verbosity levels for text coverage reports.
typedef enum {
  MOORE_TEXT_REPORT_SUMMARY = 0,  ///< Only overall summary
  MOORE_TEXT_REPORT_NORMAL = 1,   ///< Covergroups and coverpoints
  MOORE_TEXT_REPORT_DETAILED = 2  ///< Include all bins
} MooreTextReportVerbosity;

/// Generate a text coverage report file.
/// Creates a simple text file with coverage information suitable for CI parsing.
///
/// @param filename Path to the output text file
/// @param verbosity Level of detail (0=summary, 1=normal, 2=detailed)
/// @return 0 on success, non-zero on failure
int32_t __moore_coverage_report_text(const char *filename, int32_t verbosity);

/// Get a coverage summary string.
/// Returns a dynamically allocated string containing a brief coverage summary.
/// The caller is responsible for freeing the returned string with __moore_free.
///
/// @return Dynamically allocated summary string, or NULL on failure
char *__moore_coverage_report_summary(void);

/// Print a coverage summary to stdout.
/// Outputs a brief coverage summary directly to standard output.
void __moore_coverage_print_summary(void);

/// Generate a text coverage report to a string.
/// Returns a dynamically allocated string containing the full report.
/// The caller is responsible for freeing the returned string with __moore_free.
///
/// @param verbosity Level of detail (0=summary, 1=normal, 2=detailed)
/// @return Dynamically allocated report string, or NULL on failure
char *__moore_coverage_get_text_report(int32_t verbosity);

/// Print a text coverage report to stdout.
/// Outputs the coverage report directly to standard output.
///
/// @param verbosity Level of detail (0=summary, 1=normal, 2=detailed)
void __moore_coverage_print_text(int32_t verbosity);

/// Print a formatted coverage report at simulation finish.
/// This function is designed to be called at $finish to provide a summary
/// of all coverage results with pass/fail status based on goals.
///
/// @param verbosity Level of detail (-1=auto, 0=summary, 1=normal, 2=detailed)
///                  When -1, auto-selects based on number of covergroups
void __moore_coverage_report_on_finish(int32_t verbosity);

//===----------------------------------------------------------------------===//
// Coverage Database Save/Load/Merge Operations
//===----------------------------------------------------------------------===//
//
// These functions support coverage database persistence and merging.
// This enables verification flows that combine coverage from multiple
// simulation runs: run1.db + run2.db + run3.db -> merged.db
//
// The database format is JSON-based for interoperability with other tools.
//

/// Opaque handle to a coverage database.
/// Used for loading, merging, and manipulating coverage data from files.
typedef struct MooreCoverageDB *MooreCoverageDBHandle;

/// Save all current coverage data to a database file.
/// Writes all registered covergroups and their data to a JSON file.
///
/// @param filename Path to the output file (null-terminated string)
/// @return 0 on success, non-zero on failure
int32_t __moore_coverage_save(const char *filename);

/// Load a coverage database from a file.
/// Creates a new coverage database handle that can be used for merging.
/// The returned handle must be freed with __moore_coverage_db_free.
///
/// @param filename Path to the input file (null-terminated string)
/// @return Handle to the loaded database, or NULL on failure
MooreCoverageDBHandle __moore_coverage_load(const char *filename);

/// Free a coverage database handle.
/// Releases all resources associated with a loaded coverage database.
///
/// @param db Handle to the database to free
void __moore_coverage_db_free(MooreCoverageDBHandle db);

/// Merge a loaded coverage database into the current coverage state.
/// Combines bin hit counts and value tracking data from the loaded database
/// with the current registered covergroups. Covergroups and coverpoints are
/// matched by name.
///
/// @param db Handle to the database to merge
/// @return 0 on success, non-zero on failure
int32_t __moore_coverage_merge(MooreCoverageDBHandle db);

/// Merge coverage from a file directly into the current state.
/// Convenience function that combines load and merge operations.
///
/// @param filename Path to the coverage database file
/// @return 0 on success, non-zero on failure
int32_t __moore_coverage_merge_file(const char *filename);

/// Merge two coverage database files into a new output file.
/// Creates a new database containing the combined coverage from both inputs.
/// Neither input is modified.
///
/// @param file1 Path to the first input database
/// @param file2 Path to the second input database
/// @param output Path to the output merged database
/// @return 0 on success, non-zero on failure
int32_t __moore_coverage_merge_files(const char *file1, const char *file2,
                                      const char *output);

/// Get the number of covergroups in a loaded database.
///
/// @param db Handle to the database
/// @return Number of covergroups, or -1 on error
int32_t __moore_coverage_db_get_num_covergroups(MooreCoverageDBHandle db);

/// Get the name of a covergroup in a loaded database.
///
/// @param db Handle to the database
/// @param index Index of the covergroup
/// @return Name of the covergroup, or NULL on error
const char *__moore_coverage_db_get_covergroup_name(MooreCoverageDBHandle db,
                                                     int32_t index);

/// Get coverage percentage from a loaded database.
///
/// @param db Handle to the database
/// @param cg_name Name of the covergroup (NULL for total coverage)
/// @return Coverage percentage, or -1.0 on error
double __moore_coverage_db_get_coverage(MooreCoverageDBHandle db,
                                         const char *cg_name);

//===----------------------------------------------------------------------===//
// Coverage Database Persistence with Metadata
//===----------------------------------------------------------------------===//
//
// Enhanced coverage database functions that include metadata such as test name,
// timestamp, and other information for tracking coverage across multiple runs.
//

/// Metadata structure for coverage database.
/// Contains information about the test run that generated the coverage data.
typedef struct {
  const char *test_name;     ///< Name of the test that generated this coverage
  int64_t timestamp;         ///< Unix timestamp when coverage was saved
  const char *simulator;     ///< Name of the simulator (if known)
  const char *version;       ///< Database format version
  const char *comment;       ///< Optional user comment
} MooreCoverageMetadata;

/// Save coverage database with metadata.
/// Includes test name, timestamp, and other metadata in the saved file.
///
/// @param filename Path to the output file (null-terminated string)
/// @param test_name Name of the test (can be NULL)
/// @param comment Optional comment (can be NULL)
/// @return 0 on success, non-zero on failure
int32_t __moore_coverage_save_db(const char *filename, const char *test_name,
                                  const char *comment);

/// Load coverage database with metadata.
/// Loads a coverage database and makes it available for merging.
/// The returned handle must be freed with __moore_coverage_db_free.
///
/// @param filename Path to the input file (null-terminated string)
/// @return Handle to the loaded database, or NULL on failure
MooreCoverageDBHandle __moore_coverage_load_db(const char *filename);

/// Merge another coverage database file into the current state.
/// Loads the file, merges its data into all matching covergroups, then frees it.
/// This is a convenience function combining load and merge operations.
///
/// @param filename Path to the coverage database file to merge
/// @return 0 on success, non-zero on failure
int32_t __moore_coverage_merge_db(const char *filename);

/// Get metadata from a loaded coverage database.
/// Returns a pointer to the metadata structure. The pointer is valid
/// until the database handle is freed.
///
/// @param db Handle to the loaded database
/// @return Pointer to metadata, or NULL if not available
const MooreCoverageMetadata *__moore_coverage_db_get_metadata(
    MooreCoverageDBHandle db);

/// Set global test name for coverage operations.
/// This test name will be used by __moore_coverage_save_db when no explicit
/// test name is provided.
///
/// @param test_name Global test name (will be copied)
void __moore_coverage_set_test_name(const char *test_name);

/// Get the currently set global test name.
///
/// @return Current global test name, or NULL if not set
const char *__moore_coverage_get_test_name(void);

//===----------------------------------------------------------------------===//
// UCDB-Compatible Coverage File Format
//===----------------------------------------------------------------------===//
//
// These functions provide UCDB (Unified Coverage Database) compatible file
// format support for coverage data persistence. While not a true UCDB binary
// format, the JSON schema is designed to be compatible with UCDB semantics
// and can be easily converted to/from UCDB format.
//
// The UCDB-like JSON format includes:
// - Schema version for forward/backward compatibility
// - Rich metadata (tool info, timestamps, test names, user attributes)
// - Hierarchical covergroup/coverpoint/bin structure
// - Support for all bin types (value, range, wildcard, transition)
// - Merge history tracking for multi-run coverage accumulation
// - Cross coverage data (future)
//
// Format Version History:
//   2.0 - UCDB-compatible JSON format with enhanced metadata
//   1.1 - Added metadata section (test_name, timestamp, comment)
//   1.0 - Basic coverage report format
//

/// Current UCDB-compatible format version.
#define MOORE_UCDB_FORMAT_VERSION "2.0"

/// UCDB format identifier magic string.
#define MOORE_UCDB_FORMAT_MAGIC "circt-ucdb"

/// Extended metadata structure for UCDB-compatible databases.
/// Contains comprehensive information about the coverage collection session.
typedef struct {
  const char *format_version;    ///< UCDB format version (e.g., "2.0")
  const char *schema_id;         ///< Schema identifier (MOORE_UCDB_FORMAT_MAGIC)
  const char *test_name;         ///< Name of the test/testbench
  const char *test_seed;         ///< Random seed used (if applicable)
  int64_t start_time;            ///< Simulation start timestamp (Unix epoch)
  int64_t end_time;              ///< Simulation end timestamp (Unix epoch)
  int64_t sim_time;              ///< Simulation time (in time units)
  const char *time_unit;         ///< Time unit (ns, ps, etc.)
  const char *tool_name;         ///< Tool/simulator name
  const char *tool_version;      ///< Tool/simulator version
  const char *hostname;          ///< Machine hostname
  const char *username;          ///< User who ran the simulation
  const char *workdir;           ///< Working directory
  const char *command_line;      ///< Command line used
  const char *comment;           ///< User-provided comment
  int32_t num_merged_runs;       ///< Number of merged coverage runs
  const char **merged_test_names; ///< Names of merged tests (array)
  int32_t num_user_attrs;        ///< Number of user attributes
  const char **user_attr_names;  ///< User attribute names (array)
  const char **user_attr_values; ///< User attribute values (array)
} MooreUCDBMetadata;

/// Write coverage data in UCDB-compatible JSON format.
/// Creates a comprehensive coverage database file with full metadata,
/// hierarchical structure, and all bin details.
///
/// The output format follows UCDB semantics:
/// - covergroups contain coverpoints and crosses
/// - coverpoints contain bins with type, kind, and hit counts
/// - All metadata is preserved for merge and analysis
///
/// @param filename Path to the output file (null-terminated string)
/// @param metadata Extended metadata for the database (can be NULL for defaults)
/// @return 0 on success, non-zero on failure
int32_t __moore_coverage_write_ucdb(const char *filename,
                                     const MooreUCDBMetadata *metadata);

/// Read coverage data from UCDB-compatible JSON format.
/// Loads a coverage database with full metadata preservation.
/// The returned handle must be freed with __moore_coverage_db_free.
///
/// This function can read:
/// - UCDB-compatible JSON files (version 2.0+)
/// - Legacy coverage database files (version 1.x)
///
/// @param filename Path to the input file (null-terminated string)
/// @return Handle to the loaded database, or NULL on failure
MooreCoverageDBHandle __moore_coverage_read_ucdb(const char *filename);

/// Get extended UCDB metadata from a loaded database.
/// Returns a pointer to the extended metadata structure.
/// The pointer is valid until the database handle is freed.
///
/// @param db Handle to the loaded database
/// @return Pointer to extended metadata, or NULL if not available/not UCDB format
const MooreUCDBMetadata *__moore_coverage_db_get_ucdb_metadata(
    MooreCoverageDBHandle db);

/// Merge multiple UCDB-compatible coverage files.
/// Creates a new database containing combined coverage from all input files.
/// Tracks merge history in the output metadata.
///
/// @param input_files Array of input file paths
/// @param num_files Number of input files
/// @param output_file Path to the output merged database
/// @param comment Optional comment for the merge operation (can be NULL)
/// @return 0 on success, non-zero on failure
int32_t __moore_coverage_merge_ucdb_files(const char **input_files,
                                           int32_t num_files,
                                           const char *output_file,
                                           const char *comment);

/// Check if a file is in UCDB-compatible format.
/// Reads the file header to determine if it's a valid UCDB-like file.
///
/// @param filename Path to the file to check
/// @return 1 if UCDB-compatible format, 0 if not, -1 on error
int32_t __moore_coverage_is_ucdb_format(const char *filename);

/// Get the format version of a coverage database file.
/// Returns the version string from the file without fully loading it.
///
/// @param filename Path to the file to check
/// @return Version string (caller must NOT free), or NULL on error
const char *__moore_coverage_get_file_version(const char *filename);

/// Set a user attribute on the current coverage session.
/// User attributes are included in UCDB output and preserved across merges.
///
/// @param name Attribute name (will be copied)
/// @param value Attribute value (will be copied)
void __moore_coverage_set_user_attr(const char *name, const char *value);

/// Get a user attribute from the current coverage session.
///
/// @param name Attribute name to look up
/// @return Attribute value, or NULL if not found
const char *__moore_coverage_get_user_attr(const char *name);

//===----------------------------------------------------------------------===//
// Constraint Solving Operations
//===----------------------------------------------------------------------===//
//
// These functions provide basic constraint-aware randomization support.
// They are stubs for future Z3/SMT solver integration. Currently they
// implement simple bounds-based randomization without full constraint solving.
//

/// Check if a value satisfies a range constraint.
/// @param value The value to check
/// @param min Minimum allowed value (inclusive)
/// @param max Maximum allowed value (inclusive)
/// @return 1 if value is within [min, max], 0 otherwise
int __moore_constraint_check_range(int64_t value, int64_t min, int64_t max);

/// Randomize with a range constraint.
/// Generates a random value within the specified range [min, max].
/// @param min Minimum value (inclusive)
/// @param max Maximum value (inclusive)
/// @return A random value in the range [min, max]
int64_t __moore_randomize_with_range(int64_t min, int64_t max);

/// Randomize with a modulo constraint.
/// Generates a random value that satisfies: value % mod == remainder.
/// @param mod The modulo divisor (must be positive)
/// @param remainder The required remainder (must be in range [0, mod-1])
/// @return A random value satisfying the modulo constraint
int64_t __moore_randomize_with_modulo(int64_t mod, int64_t remainder);

/// Randomize with multiple range constraints.
/// Generates a random value that falls within one of the given ranges.
/// @param ranges Array of range pairs [min1, max1, min2, max2, ...]
/// @param numRanges Number of range pairs (not array length)
/// @return A random value within one of the ranges
int64_t __moore_randomize_with_ranges(int64_t *ranges, int64_t numRanges);

//===----------------------------------------------------------------------===//
// Constraint Solving with Iteration Limits
//===----------------------------------------------------------------------===//
//
// These functions provide constraint solving with iteration limits and fallback
// strategies. They prevent infinite loops on unsatisfiable constraints and
// provide statistics for debugging.
//

/// Default iteration limit for constraint solving (10000 attempts).
#define MOORE_CONSTRAINT_DEFAULT_ITERATION_LIMIT 10000

/// Constraint solving result codes.
enum MooreConstraintResult {
  MOORE_CONSTRAINT_SUCCESS = 0,        ///< Constraint satisfied successfully
  MOORE_CONSTRAINT_FALLBACK = 1,       ///< Used fallback (unconstrained random)
  MOORE_CONSTRAINT_ITERATION_LIMIT = 2 ///< Hit iteration limit, used fallback
};

/// Statistics for constraint solving operations.
/// Track solving attempts, successes, and failures for debugging.
typedef struct {
  int64_t totalAttempts;      ///< Total number of solve attempts
  int64_t successfulSolves;   ///< Number of successful constraint solves
  int64_t fallbackCount;      ///< Number of times fallback was used
  int64_t iterationLimitHits; ///< Number of times iteration limit was hit
  int64_t lastIterations;     ///< Iterations used in last solve attempt
} MooreConstraintStats;

/// Get global constraint solving statistics.
/// Returns a pointer to the global statistics structure.
/// Thread-safe: uses atomic operations for counters.
/// @return Pointer to the global constraint statistics
MooreConstraintStats *__moore_constraint_get_stats(void);

/// Reset global constraint solving statistics to zero.
void __moore_constraint_reset_stats(void);

/// Set the global iteration limit for constraint solving.
/// @param limit Maximum number of iterations (0 = use default)
void __moore_constraint_set_iteration_limit(int64_t limit);

/// Get the current global iteration limit.
/// @return Current iteration limit
int64_t __moore_constraint_get_iteration_limit(void);

/// Constraint predicate function type.
/// Used for custom constraint checking during randomization.
/// @param value The value to check
/// @param userData User-provided context data
/// @return true if the constraint is satisfied, false otherwise
typedef bool (*MooreConstraintPredicate)(int64_t value, void *userData);

/// Randomize with a custom constraint predicate and iteration limit.
/// Attempts to find a value within [min, max] that satisfies the predicate.
/// Falls back to unconstrained random if constraint cannot be satisfied.
///
/// @param min Minimum value (inclusive)
/// @param max Maximum value (inclusive)
/// @param predicate Function to check if value satisfies constraints
/// @param userData User data passed to predicate function
/// @param iterationLimit Maximum solve attempts (0 = use global default)
/// @param resultOut Pointer to store result code (can be NULL)
/// @return A value that satisfies the constraint, or fallback random value
int64_t __moore_randomize_with_constraint(int64_t min, int64_t max,
                                          MooreConstraintPredicate predicate,
                                          void *userData, int64_t iterationLimit,
                                          int32_t *resultOut);

/// Randomize with multiple range constraints and iteration limit.
/// Attempts to find a value within one of the ranges that satisfies
/// an optional predicate. Falls back to unconstrained random if needed.
///
/// @param ranges Array of range pairs [min1, max1, min2, max2, ...]
/// @param numRanges Number of range pairs
/// @param predicate Optional predicate for additional constraints (can be NULL)
/// @param userData User data for predicate
/// @param iterationLimit Maximum solve attempts (0 = use global default)
/// @param resultOut Pointer to store result code (can be NULL)
/// @return A value satisfying constraints, or fallback random value
int64_t __moore_randomize_with_ranges_constrained(int64_t *ranges,
                                                   int64_t numRanges,
                                                   MooreConstraintPredicate predicate,
                                                   void *userData,
                                                   int64_t iterationLimit,
                                                   int32_t *resultOut);

/// Report a constraint solving warning to stderr.
/// Called when constraint solving hits iteration limit or uses fallback.
/// @param message Description of the constraint issue
/// @param iterations Number of iterations attempted
/// @param variableName Name of the constrained variable (can be NULL)
void __moore_constraint_warn(const char *message, int64_t iterations,
                             const char *variableName);

/// Enable or disable constraint solving warnings.
/// @param enabled true to enable warnings, false to suppress them
void __moore_constraint_set_warnings_enabled(bool enabled);

/// Check if constraint solving warnings are enabled.
/// @return true if warnings are enabled, false otherwise
bool __moore_constraint_warnings_enabled(void);

//===----------------------------------------------------------------------===//
// Pre/Post Randomize Callbacks
//===----------------------------------------------------------------------===//
//
// SystemVerilog supports pre_randomize() and post_randomize() callback methods
// that are invoked before and after randomization respectively.
// IEEE 1800-2017 Section 18.6.1 "Pre and post randomize methods".
//

/// Call pre_randomize() callback on a class object.
/// This is called before the randomization process begins.
/// @param classPtr Pointer to the class instance
///
/// NOTE: The MooreToCore lowering now generates direct calls to user-defined
/// pre_randomize methods. This runtime function is a fallback stub.
void __moore_call_pre_randomize(void *classPtr);

/// Call post_randomize() callback on a class object.
/// This is called after randomization succeeds.
/// @param classPtr Pointer to the class instance
///
/// NOTE: The MooreToCore lowering now generates direct calls to user-defined
/// post_randomize methods. This runtime function is a fallback stub.
void __moore_call_post_randomize(void *classPtr);

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

/// Get the current constraint mode (1 = enabled, 0 = disabled).
/// Returns 1 if the constraint has not been explicitly disabled.
/// @param classPtr Pointer to the class instance
/// @param constraintName Name of the constraint (NULL for class-level query)
/// @return Current mode (0 = disabled, 1 = enabled)
int32_t __moore_constraint_mode_get(void *classPtr, const char *constraintName);

/// Set the constraint mode and return the previous mode.
/// @param classPtr Pointer to the class instance
/// @param constraintName Name of the constraint (NULL for class-level)
/// @param mode New mode: 0 = disable, 1 = enable
/// @return Previous mode value
int32_t __moore_constraint_mode_set(void *classPtr, const char *constraintName,
                                    int32_t mode);

/// Disable all constraints on a class object.
/// @param classPtr Pointer to the class instance
/// @return 1 if any constraints were enabled, 0 otherwise
int32_t __moore_constraint_mode_disable_all(void *classPtr);

/// Enable all constraints on a class object.
/// @param classPtr Pointer to the class instance
/// @return 1 if any constraints were disabled, 0 otherwise
int32_t __moore_constraint_mode_enable_all(void *classPtr);

/// Check if a specific constraint is enabled.
/// Takes into account both individual constraint mode and "disable all" flag.
/// @param classPtr Pointer to the class instance
/// @param constraintName Name of the constraint to check
/// @return 1 if enabled, 0 if disabled
int32_t __moore_is_constraint_enabled(void *classPtr,
                                      const char *constraintName);

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

/// Check if an implication constraint is satisfied.
/// Implements the SystemVerilog implication operator: antecedent -> consequent
/// If antecedent is false, the implication is trivially true.
/// If antecedent is true, the consequent must be true.
///
/// @param antecedent The condition that triggers the implication (0 or 1)
/// @param consequent The constraint that must hold when antecedent is true (0 or 1)
/// @return 1 if implication is satisfied, 0 otherwise
int32_t __moore_constraint_check_implication(int32_t antecedent,
                                              int32_t consequent);

/// Check a nested implication constraint (a -> (b -> c)).
/// Evaluates nested implications from left to right.
/// Returns 1 if the entire nested implication chain is satisfied.
///
/// @param outer The outer antecedent
/// @param inner The inner antecedent (consequent of outer)
/// @param consequent The final consequent
/// @return 1 if nested implication is satisfied, 0 otherwise
int32_t __moore_constraint_check_nested_implication(int32_t outer,
                                                     int32_t inner,
                                                     int32_t consequent);

/// Evaluate an implication constraint and apply soft/hard semantics.
/// Soft implications provide default behavior when the antecedent is true.
/// Hard implications are enforced strictly.
///
/// @param antecedent The condition that triggers the implication
/// @param consequentSatisfied Whether the consequent constraint is satisfied
/// @param isSoft 1 if this is a soft implication, 0 for hard
/// @return 1 if constraint passes (or soft fallback applies), 0 otherwise
int32_t __moore_constraint_check_implication_soft(int32_t antecedent,
                                                   int32_t consequentSatisfied,
                                                   int32_t isSoft);

/// Statistics tracking for implication constraint evaluation.
/// Incremented when implication constraints are checked at runtime.
typedef struct {
  int64_t totalImplications;     ///< Total implication checks performed
  int64_t triggeredImplications; ///< Implications where antecedent was true
  int64_t satisfiedImplications; ///< Implications that were satisfied
  int64_t softFallbacks;         ///< Soft implications using fallback
} MooreImplicationStats;

/// Get global implication constraint statistics.
/// @return Pointer to the global implication statistics
MooreImplicationStats *__moore_implication_get_stats(void);

/// Reset global implication statistics to zero.
void __moore_implication_reset_stats(void);

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

/// Check if all elements in an array are unique.
/// This implements the SystemVerilog `unique {arr}` constraint.
/// @param array Pointer to the array data
/// @param numElements Number of elements in the array
/// @param elementSize Size of each element in bytes
/// @return 1 if all elements are unique, 0 if duplicates exist
int32_t __moore_constraint_unique_check(void *array, int64_t numElements,
                                        int64_t elementSize);

/// Check if multiple scalar variables are all unique.
/// This implements the SystemVerilog `unique {a, b, c}` constraint.
/// @param values Pointer to array of values to check
/// @param numValues Number of values in the array
/// @param valueSize Size of each value in bytes
/// @return 1 if all values are unique, 0 if duplicates exist
int32_t __moore_constraint_unique_scalars(void *values, int64_t numValues,
                                          int64_t valueSize);

/// Randomize an array ensuring all elements are unique.
/// Generates random values for each element such that no two elements are equal.
/// @param array Pointer to the array data (modified in place)
/// @param numElements Number of elements in the array
/// @param elementSize Size of each element in bytes
/// @param minValue Minimum allowed value for elements
/// @param maxValue Maximum allowed value for elements
/// @return 1 on success, 0 if unable to generate unique values
int32_t __moore_randomize_unique_array(void *array, int64_t numElements,
                                       int64_t elementSize, int64_t minValue,
                                       int64_t maxValue);

/// Validate a foreach constraint on an array.
/// Checks that all elements satisfy a predicate function.
/// @param array Pointer to the array data
/// @param numElements Number of elements in the array
/// @param elementSize Size of each element in bytes
/// @param predicate Function to check each element
/// @param userData User data passed to predicate
/// @return 1 if all elements satisfy the predicate, 0 otherwise
int32_t __moore_constraint_foreach_validate(void *array, int64_t numElements,
                                            int64_t elementSize,
                                            MooreConstraintPredicate predicate,
                                            void *userData);

/// Validate an array size constraint.
/// Checks that the array has exactly the expected number of elements.
/// @param array Pointer to the array structure (queue or dynamic array)
/// @param expectedSize Expected number of elements
/// @return 1 if size matches, 0 otherwise
int32_t __moore_constraint_size_check(MooreQueue *array, int64_t expectedSize);

/// Validate an array sum constraint.
/// Checks that the sum of all elements equals the expected value.
/// @param array Pointer to the array structure
/// @param elementSize Size of each element in bytes
/// @param expectedSum Expected sum of all elements
/// @return 1 if sum matches, 0 otherwise
int32_t __moore_constraint_sum_check(MooreQueue *array, int64_t elementSize,
                                     int64_t expectedSum);

//===----------------------------------------------------------------------===//
// Rand Mode Control
//===----------------------------------------------------------------------===//
//
// SystemVerilog supports rand_mode() to enable/disable random variables.
// IEEE 1800-2017 Section 18.8 "Disabling random variables and constraints".
//
// rand_mode(0) disables a random variable or all variables on an object
// rand_mode(1) enables a random variable or all variables on an object
// rand_mode() returns the current mode (0 or 1)
//

/// Get the current rand mode (1 = enabled, 0 = disabled).
/// Returns 1 if the variable has not been explicitly disabled.
/// @param classPtr Pointer to the class instance
/// @param propertyName Name of the property (NULL for class-level query)
/// @return Current mode (0 = disabled, 1 = enabled)
int32_t __moore_rand_mode_get(void *classPtr, const char *propertyName);

/// Set the rand mode and return the previous mode.
/// @param classPtr Pointer to the class instance
/// @param propertyName Name of the property (NULL for class-level)
/// @param mode New mode: 0 = disable, 1 = enable
/// @return Previous mode value
int32_t __moore_rand_mode_set(void *classPtr, const char *propertyName,
                              int32_t mode);

/// Disable all random variables on a class object.
/// @param classPtr Pointer to the class instance
/// @return 1 if any variables were enabled, 0 otherwise
int32_t __moore_rand_mode_disable_all(void *classPtr);

/// Enable all random variables on a class object.
/// @param classPtr Pointer to the class instance
/// @return 1 if any variables were disabled, 0 otherwise
int32_t __moore_rand_mode_enable_all(void *classPtr);

/// Check if a specific random variable is enabled.
/// Takes into account both individual rand mode and "disable all" flag.
/// @param classPtr Pointer to the class instance
/// @param propertyName Name of the property to check
/// @return 1 if enabled, 0 if disabled
int32_t __moore_is_rand_enabled(void *classPtr, const char *propertyName);

//===----------------------------------------------------------------------===//
// File I/O Operations
//===----------------------------------------------------------------------===//
//
// These functions provide runtime support for SystemVerilog file I/O
// operations: $fopen, $fwrite, $fclose. They implement the file I/O
// system tasks/functions as specified in IEEE 1800-2017 Section 21.3.
//

/// Open a file and return a file descriptor.
/// Implements the SystemVerilog $fopen system function.
/// @param filename Pointer to the filename string structure
/// @param mode Pointer to the mode string structure (e.g., "w", "r", "a")
///        If mode is NULL, defaults to "r"
/// @return A 32-bit file descriptor (MCD), or 0 if the open fails
int32_t __moore_fopen(MooreString *filename, MooreString *mode);

/// Write a formatted string to a file.
/// Implements the SystemVerilog $fwrite system task.
/// The format string has already been evaluated by the Moore dialect's
/// format string operations; this function just writes the result to file.
/// @param fd File descriptor from $fopen
/// @param message Pointer to the message string structure to write
void __moore_fwrite(int32_t fd, MooreString *message);

/// Close a file.
/// Implements the SystemVerilog $fclose system task.
/// @param fd File descriptor to close
void __moore_fclose(int32_t fd);

/// Read a single character from a file.
/// Implements the SystemVerilog $fgetc system function.
/// @param fd File descriptor to read from
/// @return The character read, or EOF (-1) on error or end-of-file
int32_t __moore_fgetc(int32_t fd);

/// Read a line from a file into a string.
/// Implements the SystemVerilog $fgets system function.
/// @param str Pointer to the destination string structure
/// @param fd File descriptor to read from
/// @return The number of characters read, or 0 on error or end-of-file
int32_t __moore_fgets(MooreString *str, int32_t fd);

/// Check if end-of-file has been reached.
/// Implements the SystemVerilog $feof system function.
/// @param fd File descriptor to check
/// @return Non-zero if EOF has been reached, 0 otherwise
int32_t __moore_feof(int32_t fd);

/// Flush file output buffer.
/// Implements the SystemVerilog $fflush system task.
/// @param fd File descriptor to flush (0 flushes all files)
void __moore_fflush(int32_t fd);

/// Get current file position.
/// Implements the SystemVerilog $ftell system function.
/// @param fd File descriptor
/// @return Current file position, or -1 on error
int32_t __moore_ftell(int32_t fd);

//===----------------------------------------------------------------------===//
// Display System Tasks
//===----------------------------------------------------------------------===//
//
// These functions provide runtime support for SystemVerilog display system
// tasks: $display, $write, $strobe, $monitor. They implement the display
// functionality as specified in IEEE 1800-2017 Section 21.2.
//
// The format string has already been evaluated by the Moore dialect's
// format string operations; these functions output the pre-formatted result.
//
// Format specifiers supported in pre-processing:
// - %d, %0d, %8d - decimal integer (with optional width)
// - %h, %x, %0h - hexadecimal (lower/upper case)
// - %b, %0b - binary
// - %o - octal
// - %s - string
// - %t - time value
// - %m - hierarchical module path
// - %c - single character
// - %e, %f, %g - floating point formats
//

/// Display a formatted message to stdout with a trailing newline.
/// Implements the SystemVerilog $display system task.
/// The message is a pre-formatted string from format string evaluation.
/// @param message Pointer to the message string structure
void __moore_display(MooreString *message);

/// Display a formatted message to stdout without a trailing newline.
/// Implements the SystemVerilog $write system task.
/// @param message Pointer to the message string structure
void __moore_write(MooreString *message);

/// Schedule a message to be displayed at the end of the current timestep.
/// Implements the SystemVerilog $strobe system task.
/// Strobe output is postponed to the end of the current simulation time step,
/// after all processes have executed for that time step.
/// @param message Pointer to the message string structure
void __moore_strobe(MooreString *message);

/// Register a message to be displayed when monitored values change.
/// Implements the SystemVerilog $monitor system task.
/// Monitor continuously displays its arguments whenever any of the listed
/// arguments change value (except $time, $stime, $realtime).
/// Only one $monitor can be active at a time.
/// @param message Pointer to the message string structure
/// @param values Array of pointers to monitored values (for change detection)
/// @param numValues Number of values being monitored
/// @param valueSizes Array of sizes (in bytes) for each monitored value
void __moore_monitor(MooreString *message, void **values, int32_t numValues,
                     int32_t *valueSizes);

/// Disable $monitor output.
/// Implements the SystemVerilog $monitoroff system task.
void __moore_monitoroff(void);

/// Enable $monitor output (default state).
/// Implements the SystemVerilog $monitoron system task.
void __moore_monitoron(void);

/// Print a dynamic MooreString directly to stdout (for sim::FormatDynStringOp).
/// This is called when the LLVM lowering encounters a dynamic string value
/// that needs to be printed as part of a display task.
/// @param str Pointer to the dynamic string structure
void __moore_print_dyn_string(MooreString *str);

/// Get the current simulation time.
/// Implements the SystemVerilog $time system function.
/// @return The current simulation time as a 64-bit value
int64_t __moore_get_time(void);

/// Set the current simulation time (called by the simulation scheduler).
/// @param time The new simulation time
void __moore_set_time(int64_t time);

/// Execute pending strobe callbacks at end of timestep.
/// Called by the simulation scheduler at the end of each timestep.
void __moore_strobe_flush(void);

/// Check and execute monitor callback if values changed.
/// Called by the simulation scheduler after each delta cycle.
void __moore_monitor_check(void);

//===----------------------------------------------------------------------===//
// Simulation Control Tasks
//===----------------------------------------------------------------------===//
//
// These functions implement the SystemVerilog simulation control and severity
// reporting tasks: $finish, $fatal, $error, $warning, $info.
//
// The severity tasks track error and warning counts during simulation, which
// can be queried at the end to determine overall simulation status.
//
// Severity Levels (from IEEE 1800-2017):
// - $fatal: Stop simulation immediately with exit code
// - $error: Report error, increment error count, continue simulation
// - $warning: Report warning, increment warning count, continue simulation
// - $info: Report informational message, continue simulation
//

/// End simulation with the specified exit code.
/// Implements the SystemVerilog $finish system task.
/// @param exit_code The exit code to return (0 = success, non-zero = failure)
void __moore_finish(int32_t exit_code);

/// Report a fatal error and stop simulation.
/// Implements the SystemVerilog $fatal system task.
/// @param exit_code The exit code to return (typically 1)
/// @param message Pointer to the message string structure (may be NULL)
void __moore_fatal(int32_t exit_code, MooreString *message);

/// Report a non-fatal error and continue simulation.
/// Implements the SystemVerilog $error system task.
/// Increments the error count.
/// @param message Pointer to the message string structure (may be NULL)
void __moore_error(MooreString *message);

/// Report a warning and continue simulation.
/// Implements the SystemVerilog $warning system task.
/// Increments the warning count.
/// @param message Pointer to the message string structure (may be NULL)
void __moore_warning(MooreString *message);

/// Report an informational message and continue simulation.
/// Implements the SystemVerilog $info system task.
/// @param message Pointer to the message string structure (may be NULL)
void __moore_info(MooreString *message);

/// Get the current error count.
/// @return Number of errors reported during simulation
int32_t __moore_get_error_count(void);

/// Get the current warning count.
/// @return Number of warnings reported during simulation
int32_t __moore_get_warning_count(void);

/// Reset error and warning counts to zero.
/// Useful for test frameworks that want to clear counts between tests.
void __moore_reset_severity_counts(void);

/// Get a summary of errors and warnings at end of simulation.
/// Prints a summary message if there were any errors or warnings.
/// @return The total number of errors (0 if no errors)
int32_t __moore_severity_summary(void);

/// Set whether $finish should actually exit or just set a flag.
/// This is useful for testing the runtime functions.
/// @param should_exit true to exit on $finish, false to just set flag
void __moore_set_finish_exits(bool should_exit);

/// Check if $finish has been called.
/// @return true if $finish or $fatal was called
bool __moore_finish_called(void);

/// Get the exit code from the last $finish or $fatal call.
/// @return The exit code (0 if $finish not called)
int32_t __moore_get_exit_code(void);

/// Reset the finish state (for testing purposes).
void __moore_reset_finish_state(void);

//===----------------------------------------------------------------------===//
// Memory Management
//===----------------------------------------------------------------------===//

/// Free memory allocated by the Moore runtime.
/// Should be called to release strings and arrays when no longer needed.
/// @param ptr Pointer to the memory to free
void __moore_free(void *ptr);

//===----------------------------------------------------------------------===//
// UVM Coverage Model API
//===----------------------------------------------------------------------===//
//
// These functions provide UVM-compatible coverage API for register and field
// coverage tracking. They implement the uvm_coverage_model_e semantics from
// IEEE 1800.2 UVM standard, enabling coverage collection tied to register
// and field access patterns.
//
// UVM Coverage Models (from uvm_reg_model.svh):
// - UVM_CVR_REG_BITS: Coverage of individual register bits
// - UVM_CVR_ADDR_MAP: Coverage of address map accesses
// - UVM_CVR_FIELD_VALS: Coverage of field value ranges
// - UVM_CVR_ALL: All coverage models enabled
//
// Usage:
//   __moore_uvm_set_coverage_model(UVM_CVR_REG_BITS | UVM_CVR_FIELD_VALS);
//   __moore_uvm_coverage_sample_reg("my_reg", value);
//   __moore_uvm_coverage_sample_field("my_field", value);
//

/// UVM coverage model enum values (matches uvm_coverage_model_e).
/// These can be combined using bitwise OR.
enum MooreUvmCoverageModel {
  UVM_CVR_REG_BITS = (1 << 0),   ///< Individual register bit coverage
  UVM_CVR_ADDR_MAP = (1 << 1),   ///< Address map access coverage
  UVM_CVR_FIELD_VALS = (1 << 2), ///< Field value range coverage
  UVM_NO_COVERAGE = 0,           ///< No coverage enabled
  UVM_CVR_ALL = ((1 << 3) - 1)   ///< All coverage models enabled
};

/// Set the UVM coverage model.
/// Specifies which coverage models are active for register/field sampling.
/// Multiple models can be combined using bitwise OR.
///
/// @param model Bitmask of MooreUvmCoverageModel values
void __moore_uvm_set_coverage_model(int32_t model);

/// Get the current UVM coverage model.
///
/// @return Bitmask of currently active MooreUvmCoverageModel values
int32_t __moore_uvm_get_coverage_model(void);

/// Check if a specific coverage model is enabled.
///
/// @param model The coverage model to check
/// @return true if the model is enabled
bool __moore_uvm_has_coverage(int32_t model);

/// Sample register access for UVM coverage.
/// Records register access for coverage tracking. Coverage is only recorded
/// if UVM_CVR_REG_BITS is enabled in the current coverage model.
/// Integrates with the existing covergroup infrastructure.
///
/// @param reg_name Name of the register being accessed
/// @param value Value being read/written
void __moore_uvm_coverage_sample_reg(const char *reg_name, int64_t value);

/// Sample field access for UVM coverage.
/// Records field access for coverage tracking. Coverage is only recorded
/// if UVM_CVR_FIELD_VALS is enabled in the current coverage model.
/// Integrates with the existing covergroup infrastructure.
///
/// @param field_name Name of the field being accessed
/// @param value Value being read/written
void __moore_uvm_coverage_sample_field(const char *field_name, int64_t value);

/// Sample address map access for UVM coverage.
/// Records address map access for coverage tracking. Coverage is only recorded
/// if UVM_CVR_ADDR_MAP is enabled in the current coverage model.
///
/// @param map_name Name of the address map
/// @param address Address being accessed
/// @param is_read true for read access, false for write access
void __moore_uvm_coverage_sample_addr_map(const char *map_name, int64_t address,
                                          bool is_read);

/// Get register coverage percentage.
/// Returns the coverage percentage for a specific register.
/// Creates an implicit covergroup for the register if it doesn't exist.
///
/// @param reg_name Name of the register
/// @return Coverage percentage (0.0 to 100.0)
double __moore_uvm_get_reg_coverage(const char *reg_name);

/// Get field coverage percentage.
/// Returns the coverage percentage for a specific field.
/// Creates an implicit covergroup for the field if it doesn't exist.
///
/// @param field_name Name of the field
/// @return Coverage percentage (0.0 to 100.0)
double __moore_uvm_get_field_coverage(const char *field_name);

/// Get total UVM register model coverage.
/// Returns the aggregate coverage across all sampled registers and fields.
///
/// @return Total coverage percentage (0.0 to 100.0)
double __moore_uvm_get_coverage(void);

/// Reset all UVM coverage data.
/// Clears all register and field coverage data while preserving the structure.
void __moore_uvm_reset_coverage(void);

/// UVM register coverage callback function type.
/// Called when a register is sampled for coverage.
typedef void (*MooreUvmRegCoverageCallback)(const char *reg_name,
                                            int64_t value,
                                            void *userData);

/// UVM field coverage callback function type.
/// Called when a field is sampled for coverage.
typedef void (*MooreUvmFieldCoverageCallback)(const char *field_name,
                                              int64_t value,
                                              void *userData);

/// Register a callback for register coverage sampling.
/// The callback is invoked each time __moore_uvm_coverage_sample_reg is called.
///
/// @param callback Function to call on register sampling (NULL to disable)
/// @param userData User data passed to the callback
void __moore_uvm_set_reg_coverage_callback(MooreUvmRegCoverageCallback callback,
                                           void *userData);

/// Register a callback for field coverage sampling.
/// The callback is invoked each time __moore_uvm_coverage_sample_field is called.
///
/// @param callback Function to call on field sampling (NULL to disable)
/// @param userData User data passed to the callback
void __moore_uvm_set_field_coverage_callback(MooreUvmFieldCoverageCallback callback,
                                             void *userData);

/// Set the coverage sample bit width for a register.
/// Specifies how many bits are tracked for UVM_CVR_REG_BITS coverage.
/// Default is 64 bits.
///
/// @param reg_name Name of the register
/// @param bit_width Number of bits to track (1-64)
void __moore_uvm_set_reg_bit_width(const char *reg_name, int32_t bit_width);

/// Set the coverage sample range for a field.
/// Specifies the value range for UVM_CVR_FIELD_VALS coverage.
/// Coverage tracks how many values in [min_val, max_val] have been seen.
///
/// @param field_name Name of the field
/// @param min_val Minimum value of the field range
/// @param max_val Maximum value of the field range
void __moore_uvm_set_field_range(const char *field_name, int64_t min_val,
                                 int64_t max_val);

//===----------------------------------------------------------------------===//
// DPI-C Import Stubs for UVM Support
//===----------------------------------------------------------------------===//
//
// These functions provide stub implementations of UVM DPI-C imports.
// They allow UVM code to compile and run without requiring external C
// libraries. The stubs provide simplified behavior suitable for basic
// testing and development.
//
// Categories:
// - HDL Access: uvm_hdl_* functions for signal manipulation
// - Regex: uvm_re_* functions for pattern matching
// - Command Line: uvm_dpi_* functions for tool information
//

//===----------------------------------------------------------------------===//
// Signal Registry Bridge
//===----------------------------------------------------------------------===//
//
// The Signal Registry Bridge connects DPI/VPI functions to actual simulation
// signals managed by the ProcessScheduler. This enables UVM HDL access
// functions like uvm_hdl_read() to return actual signal values from the
// simulation.
//
// Architecture:
// 1. LLHDProcessInterpreter registers signals with hierarchical names
// 2. Signal names are exported via __moore_signal_registry_register()
// 3. DPI functions query the registry to get SignalIds
// 4. SignalIds are used to access values via ProcessScheduler
//
// Usage:
// - During simulation initialization, register all signals with their
//   hierarchical paths using __moore_signal_registry_register()
// - Set the accessor callbacks via __moore_signal_registry_set_accessor()
// - DPI functions will automatically use the registry when available
//

/// Opaque handle to a signal in the registry
typedef uint64_t MooreSignalHandle;

/// Invalid signal handle constant
#define MOORE_INVALID_SIGNAL_HANDLE 0

/// Callback type for reading a signal value.
/// @param signalHandle The signal handle from the registry
/// @param userData User-provided context
/// @return The current signal value
typedef int64_t (*MooreSignalReadCallback)(MooreSignalHandle signalHandle,
                                           void *userData);

/// Callback type for writing/depositing a signal value.
/// @param signalHandle The signal handle from the registry
/// @param value The value to write
/// @param userData User-provided context
/// @return 1 on success, 0 on failure
typedef int32_t (*MooreSignalWriteCallback)(MooreSignalHandle signalHandle,
                                            int64_t value, void *userData);

/// Callback type for forcing a signal value.
/// @param signalHandle The signal handle from the registry
/// @param value The value to force
/// @param userData User-provided context
/// @return 1 on success, 0 on failure
typedef int32_t (*MooreSignalForceCallback)(MooreSignalHandle signalHandle,
                                            int64_t value, void *userData);

/// Callback type for releasing a forced signal.
/// @param signalHandle The signal handle from the registry
/// @param userData User-provided context
/// @return 1 on success, 0 on failure
typedef int32_t (*MooreSignalReleaseCallback)(MooreSignalHandle signalHandle,
                                              void *userData);

/// Register a signal with its hierarchical path.
/// This function is called by the simulation infrastructure to populate the
/// signal registry during initialization.
///
/// @param path Hierarchical path (e.g., "top.dut.clk")
/// @param signalHandle The signal handle (typically SignalId from ProcessScheduler)
/// @param width Bit width of the signal
/// @return 1 on success, 0 if path is invalid
int32_t __moore_signal_registry_register(const char *path,
                                         MooreSignalHandle signalHandle,
                                         uint32_t width);

/// Set the accessor callbacks for the signal registry.
/// This connects the DPI functions to the actual signal value access.
///
/// @param readCallback Callback for reading signal values
/// @param writeCallback Callback for writing/depositing signal values
/// @param forceCallback Callback for forcing signal values
/// @param releaseCallback Callback for releasing forced signals
/// @param userData User data passed to all callbacks
void __moore_signal_registry_set_accessor(MooreSignalReadCallback readCallback,
                                          MooreSignalWriteCallback writeCallback,
                                          MooreSignalForceCallback forceCallback,
                                          MooreSignalReleaseCallback releaseCallback,
                                          void *userData);

/// Look up a signal handle by hierarchical path.
///
/// @param path Hierarchical path to look up
/// @return Signal handle, or MOORE_INVALID_SIGNAL_HANDLE if not found
MooreSignalHandle __moore_signal_registry_lookup(const char *path);

/// Check if a signal path exists in the registry.
///
/// @param path Hierarchical path to check
/// @return 1 if path exists in registry, 0 otherwise
int32_t __moore_signal_registry_exists(const char *path);

/// Get the bit width of a registered signal.
///
/// @param path Hierarchical path
/// @return Bit width, or 0 if path not found
uint32_t __moore_signal_registry_get_width(const char *path);

/// Clear all registered signals from the registry.
/// This is useful for resetting between simulation runs.
void __moore_signal_registry_clear(void);

/// Get the number of registered signals.
///
/// @return Number of signals in the registry
uint64_t __moore_signal_registry_count(void);

/// Check if the signal registry is connected (has accessor callbacks set).
///
/// @return 1 if connected to actual simulation, 0 if using stub mode
int32_t __moore_signal_registry_is_connected(void);

//===----------------------------------------------------------------------===//
// Signal Registry - Hierarchy Traversal and Force/Release
//===----------------------------------------------------------------------===//

/// Look up a signal handle supporting hierarchical paths and wildcards.
/// This function tries various path formats including:
/// - Direct path lookup
/// - Partial path matching from end (e.g., "sig" matches "top.inst.sig")
/// - Array index parsing (e.g., "mem[5]")
///
/// @param path Hierarchical path to look up
/// @return Signal handle, or MOORE_INVALID_SIGNAL_HANDLE if not found
MooreSignalHandle __moore_signal_registry_lookup_hierarchical(const char *path);

/// Get a list of all registered signal paths.
/// Fills a buffer with null-separated path strings.
///
/// @param buffer Buffer to fill with null-separated paths (can be NULL)
/// @param bufferSize Size of the buffer in bytes
/// @return Number of registered signals
uint64_t __moore_signal_registry_get_paths(char *buffer, uint64_t bufferSize);

/// Check if a signal is currently forced via DPI.
///
/// @param path Hierarchical path to check
/// @return 1 if signal is forced, 0 otherwise
int32_t __moore_signal_registry_is_forced(const char *path);

/// Get the forced value for a signal (if forced).
///
/// @param path Hierarchical path
/// @param value Pointer to store the forced value
/// @return 1 if signal is forced and value was retrieved, 0 otherwise
int32_t __moore_signal_registry_get_forced_value(const char *path,
                                                  int64_t *value);

/// Set a signal as forced with a specific value.
/// This tracks the force state for DPI force/release semantics.
///
/// @param path Hierarchical path
/// @param handle Signal handle from registry
/// @param value Value to force
/// @return 1 on success, 0 on failure
int32_t __moore_signal_registry_set_forced(const char *path,
                                            MooreSignalHandle handle,
                                            int64_t value);

/// Clear the forced state for a signal.
///
/// @param path Hierarchical path
/// @return 1 if signal was forced and is now released, 0 otherwise
int32_t __moore_signal_registry_clear_forced(const char *path);

/// Clear all forced signals.
/// This is useful for resetting between simulation runs.
void __moore_signal_registry_clear_all_forced(void);

//===----------------------------------------------------------------------===//
// HDL Access Stubs (IEEE 1800.2-2017 DPI)
//===----------------------------------------------------------------------===//

/// HDL data type used for deposit/force/read operations.
/// Large enough to hold a 64-bit value.
typedef int64_t uvm_hdl_data_t;

/// Check if an HDL path exists in the design hierarchy.
/// @param path Pointer to the path string structure
/// @return 1 if path exists, 0 otherwise (stub: always returns 1)
int32_t uvm_hdl_check_path(MooreString *path);

/// Deposit a value to an HDL object (procedural assignment).
/// @param path Pointer to the path string structure
/// @param value The value to deposit
/// @return 1 on success, 0 on failure (stub: always returns 1)
int32_t uvm_hdl_deposit(MooreString *path, uvm_hdl_data_t value);

/// Force a value onto an HDL object (continuous assignment).
/// @param path Pointer to the path string structure
/// @param value The value to force
/// @return 1 on success, 0 on failure (stub: always returns 1)
int32_t uvm_hdl_force(MooreString *path, uvm_hdl_data_t value);

/// Release a forced value and read the current value.
/// @param path Pointer to the path string structure
/// @param value Pointer to store the read value
/// @return 1 on success, 0 on failure (stub: always returns 1)
int32_t uvm_hdl_release_and_read(MooreString *path, uvm_hdl_data_t *value);

/// Release a forced value on an HDL object.
/// @param path Pointer to the path string structure
/// @return 1 on success, 0 on failure (stub: always returns 1)
int32_t uvm_hdl_release(MooreString *path);

/// Read the current value of an HDL object.
/// @param path Pointer to the path string structure
/// @param value Pointer to store the read value
/// @return 1 on success, 0 on failure (stub: always returns 1)
int32_t uvm_hdl_read(MooreString *path, uvm_hdl_data_t *value);

//===----------------------------------------------------------------------===//
// Regular Expression Stubs
//===----------------------------------------------------------------------===//

/// Compile a regular expression pattern.
/// @param pattern Pointer to the regex pattern string
/// @param deglob If non-zero, treat as glob pattern and convert to regex
/// @return Opaque handle to compiled regex (stub: returns dummy non-zero value)
void *uvm_re_comp(MooreString *pattern, int32_t deglob);

/// Execute a compiled regular expression against a string.
/// @param rexp Handle from uvm_re_comp
/// @param str Pointer to the string to match against
/// @return Match index or -1 if no match (stub: returns 0 for match)
int32_t uvm_re_exec(void *rexp, MooreString *str);

/// Free a compiled regular expression.
/// @param rexp Handle from uvm_re_comp
void uvm_re_free(void *rexp);

/// Get the buffer containing the last matched substring.
/// @return Pointer to the match buffer string (stub: returns empty string)
MooreString uvm_re_buffer(void);

/// Compile and execute a regex in one call, then free it.
/// @param pattern Pointer to the regex pattern string
/// @param str Pointer to the string to match against
/// @param deglob If non-zero, treat as glob pattern
/// @param exec_ret Pointer to store the match result (0 = match, -1 = no match)
/// @return 1 if regex was valid, 0 otherwise (stub: always returns 1)
int32_t uvm_re_compexecfree(MooreString *pattern, MooreString *str,
                            int32_t deglob, int32_t *exec_ret);

/// Convert a glob pattern to a regular expression.
/// @param glob Pointer to the glob pattern string
/// @param with_brackets If non-zero, use bracket expressions in output
/// @return A string containing the converted regex pattern
MooreString uvm_re_deglobbed(MooreString *glob, int32_t with_brackets);

//===----------------------------------------------------------------------===//
// Command Line / Tool Info Stubs
//===----------------------------------------------------------------------===//

/// Get the next command line argument.
/// @param idx Pointer to the current argument index (updated on success)
/// @return Pointer to the next argument string, or NULL if no more arguments
MooreString uvm_dpi_get_next_arg_c(int32_t init);

/// Get the name of the simulation tool.
/// @return String containing the tool name (stub: returns "CIRCT")
MooreString uvm_dpi_get_tool_name_c(void);

/// Get the version of the simulation tool.
/// @return String containing the tool version (stub: returns "1.0")
MooreString uvm_dpi_get_tool_version_c(void);

/// Parse +UVM_TESTNAME from command-line arguments.
/// This function searches through all command-line arguments (from environment
/// variables CIRCT_UVM_ARGS or UVM_ARGS) for a +UVM_TESTNAME=<name> argument.
///
/// The format is: +UVM_TESTNAME=<test_class_name>
/// For example: +UVM_TESTNAME=my_test
///
/// @return A newly allocated MooreString containing the test name if found,
///         or an empty MooreString (data=NULL, len=0) if not found.
///         The caller is responsible for freeing the returned string's data
///         using __moore_free().
MooreString __moore_uvm_get_testname_from_cmdline(void);

/// Check if +UVM_TESTNAME was specified on the command line.
/// @return 1 if +UVM_TESTNAME was found, 0 otherwise
int32_t __moore_uvm_has_cmdline_testname(void);

//===----------------------------------------------------------------------===//
// VPI Stub Support
//===----------------------------------------------------------------------===//

typedef void *vpiHandle;

typedef struct vpi_value_s {
  int32_t format;
  void *value;
} vpi_value;

vpiHandle vpi_handle_by_name(const char *name, vpiHandle scope);
int32_t vpi_get(int32_t property, vpiHandle obj);
char *vpi_get_str(int32_t property, vpiHandle obj);
int32_t vpi_get_value(vpiHandle obj, vpi_value *value);
int32_t vpi_put_value(vpiHandle obj, vpi_value *value, void *time,
                      int32_t flags);
void vpi_release_handle(vpiHandle obj);

//===----------------------------------------------------------------------===//
// UVM Configuration Database
//===----------------------------------------------------------------------===//

/// Set a value in the UVM configuration database.
/// @param context Pointer to the context (currently unused, for future hierarchy support)
/// @param instName Pointer to the instance name string data
/// @param instLen Length of the instance name string
/// @param fieldName Pointer to the field name string data
/// @param fieldLen Length of the field name string
/// @param value Pointer to the value to store
/// @param valueSize Size of the value in bytes
/// @param typeId Type identifier for type checking on retrieval
void __moore_config_db_set(void *context, const char *instName, int64_t instLen,
                           const char *fieldName, int64_t fieldLen, void *value,
                           int64_t valueSize, int32_t typeId);

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
int32_t __moore_config_db_get(void *context, const char *instName,
                              int64_t instLen, const char *fieldName,
                              int64_t fieldLen, int32_t typeId, void *outValue,
                              int64_t valueSize);

/// Check if a key exists in the configuration database.
/// This uses the same matching logic as __moore_config_db_get, so it returns
/// true if a get() would succeed (including wildcard/hierarchical matches).
/// @param instName Pointer to the instance name string data
/// @param instLen Length of the instance name string
/// @param fieldName Pointer to the field name string data
/// @param fieldLen Length of the field name string
/// @return 1 if the key exists (exact or via pattern), 0 otherwise
int32_t __moore_config_db_exists(const char *instName, int64_t instLen,
                                 const char *fieldName, int64_t fieldLen);

/// Clear all entries from the configuration database.
/// This is useful for test cleanup between test cases.
void __moore_config_db_clear(void);

//===----------------------------------------------------------------------===//
// UVM Virtual Interface Binding Runtime
//===----------------------------------------------------------------------===//
//
// These functions provide runtime support for virtual interface binding in UVM
// testbenches. Virtual interfaces connect the HVL (verification) world to the
// HDL (design) world, allowing UVM drivers and monitors to access DUT signals.
//
// Usage pattern in UVM:
//   1. Create virtual interface handle: vif_handle = __moore_vif_create(...)
//   2. Bind to interface instance: __moore_vif_bind(vif_handle, interface_ptr)
//   3. Store in config_db: config_db::set(this, "*", "vif", vif_handle)
//   4. Retrieve in driver: config_db::get(this, "", "vif", driver.vif)
//   5. Access signals: __moore_vif_get_signal(vif_handle, "data", ...)
//
// Virtual interfaces are identified by:
//   - Interface type name (e.g., "apb_if")
//   - Optional modport name (e.g., "driver")
//
// The runtime maintains a registry of virtual interface handles and their
// bindings to actual interface instances.
//

/// Opaque handle for a virtual interface instance.
typedef void *MooreVifHandle;

/// Invalid/null virtual interface handle.
#define MOORE_VIF_NULL ((MooreVifHandle)0)

/// Create a new virtual interface handle.
/// This creates an unbound virtual interface that can later be bound to an
/// actual interface instance.
///
/// @param interfaceTypeName Name of the interface type (e.g., "apb_if")
/// @param interfaceTypeNameLen Length of the interface type name string
/// @param modportName Optional modport name (NULL for full interface access)
/// @param modportNameLen Length of the modport name string (0 if no modport)
/// @return A new virtual interface handle, or MOORE_VIF_NULL on failure
MooreVifHandle __moore_vif_create(const char *interfaceTypeName,
                                  int64_t interfaceTypeNameLen,
                                  const char *modportName,
                                  int64_t modportNameLen);

/// Bind a virtual interface handle to an actual interface instance.
/// This associates the virtual interface with a concrete interface instance,
/// enabling signal access through the virtual interface.
///
/// @param vif The virtual interface handle to bind
/// @param interfaceInstance Pointer to the actual interface instance
/// @return 1 on success, 0 on failure (e.g., null handle or type mismatch)
int32_t __moore_vif_bind(MooreVifHandle vif, void *interfaceInstance);

/// Check if a virtual interface is bound to an interface instance.
///
/// @param vif The virtual interface handle to check
/// @return 1 if bound, 0 if unbound or null handle
int32_t __moore_vif_is_bound(MooreVifHandle vif);

/// Get the interface instance pointer from a virtual interface.
///
/// @param vif The virtual interface handle
/// @return Pointer to the bound interface instance, or NULL if unbound
void *__moore_vif_get_instance(MooreVifHandle vif);

/// Get a signal value from a virtual interface.
/// Reads the current value of a signal within the bound interface instance.
///
/// @param vif The virtual interface handle
/// @param signalName Name of the signal to read
/// @param signalNameLen Length of the signal name string
/// @param outValue Pointer to store the signal value
/// @param valueSize Size of the output buffer in bytes
/// @return 1 on success, 0 on failure (e.g., signal not found, vif unbound)
int32_t __moore_vif_get_signal(MooreVifHandle vif, const char *signalName,
                               int64_t signalNameLen, void *outValue,
                               int64_t valueSize);

/// Set a signal value through a virtual interface.
/// Writes a value to a signal within the bound interface instance.
///
/// @param vif The virtual interface handle
/// @param signalName Name of the signal to write
/// @param signalNameLen Length of the signal name string
/// @param value Pointer to the value to write
/// @param valueSize Size of the value in bytes
/// @return 1 on success, 0 on failure (e.g., signal not found, vif unbound)
int32_t __moore_vif_set_signal(MooreVifHandle vif, const char *signalName,
                               int64_t signalNameLen, const void *value,
                               int64_t valueSize);

/// Get a reference (pointer) to a signal within the virtual interface.
/// This allows direct access to the signal storage for efficient repeated access.
///
/// @param vif The virtual interface handle
/// @param signalName Name of the signal
/// @param signalNameLen Length of the signal name string
/// @return Pointer to the signal storage, or NULL on failure
void *__moore_vif_get_signal_ref(MooreVifHandle vif, const char *signalName,
                                 int64_t signalNameLen);

/// Get the interface type name from a virtual interface handle.
///
/// @param vif The virtual interface handle
/// @return A MooreString containing the interface type name
MooreString __moore_vif_get_type_name(MooreVifHandle vif);

/// Get the modport name from a virtual interface handle (if any).
///
/// @param vif The virtual interface handle
/// @return A MooreString containing the modport name, or empty if no modport
MooreString __moore_vif_get_modport_name(MooreVifHandle vif);

/// Compare two virtual interface handles.
/// Returns true if both handles point to the same interface instance
/// (or both are null/unbound).
///
/// @param vif1 First virtual interface handle
/// @param vif2 Second virtual interface handle
/// @return 1 if equal, 0 if not equal
int32_t __moore_vif_compare(MooreVifHandle vif1, MooreVifHandle vif2);

/// Release a virtual interface handle.
/// This unbinds the virtual interface and releases associated resources.
/// The handle should not be used after this call.
///
/// @param vif The virtual interface handle to release
void __moore_vif_release(MooreVifHandle vif);

/// Clear all virtual interface handles.
/// This is useful for test cleanup between test cases.
void __moore_vif_clear_all(void);

/// Register a signal within an interface type for virtual interface access.
/// This is called during interface elaboration to register signals that can
/// be accessed through virtual interfaces.
///
/// @param interfaceTypeName Name of the interface type
/// @param interfaceTypeNameLen Length of the interface type name
/// @param signalName Name of the signal within the interface
/// @param signalNameLen Length of the signal name
/// @param signalOffset Byte offset of the signal within the interface instance
/// @param signalSize Size of the signal in bytes
/// @return 1 on success, 0 on failure
int32_t __moore_vif_register_signal(const char *interfaceTypeName,
                                    int64_t interfaceTypeNameLen,
                                    const char *signalName,
                                    int64_t signalNameLen,
                                    int64_t signalOffset,
                                    int64_t signalSize);

/// Clear all registered interface signal information.
/// This is useful for test cleanup.
void __moore_vif_clear_registry(void);

//===----------------------------------------------------------------------===//
// UVM Component Hierarchy Support
//===----------------------------------------------------------------------===//
//
// These functions provide runtime support for UVM component hierarchy methods
// that cannot be inlined due to recursion (like get_full_name).
//
// UVM's get_full_name() is recursive:
//   virtual function string get_full_name();
//     if (m_parent == null || m_parent.get_name() == "")
//       return m_name;
//     else
//       return {m_parent.get_full_name(), ".", m_name};
//   endfunction
//
// Since recursive functions cannot be inlined (LLHD IR limitation), we provide
// an iterative runtime implementation that walks the parent chain.
//

/// Get the full hierarchical name of a UVM component.
/// This function iteratively walks the parent chain to build the full name,
/// avoiding the recursion that cannot be inlined.
///
/// The function needs to know the layout of the component class to access:
/// - m_parent: pointer to parent component (at parentOffset)
/// - m_name: the component's name string (at nameOffset)
///
/// @param component Pointer to the component instance
/// @param parentOffset Byte offset of the m_parent field within the component
/// @param nameOffset Byte offset of the m_name field (MooreString) within the component
/// @return A new MooreString containing the full hierarchical name (e.g., "top.env.agent")
MooreString __moore_component_get_full_name(void *component,
                                            int64_t parentOffset,
                                            int64_t nameOffset);

//===----------------------------------------------------------------------===//
// UVM Runtime Infrastructure
//===----------------------------------------------------------------------===//
//
// These functions provide the basic UVM runtime support needed to execute
// UVM testbenches. The implementation supports the UVM run_test() entry point
// and can be expanded to support more complex UVM features like the phase
// system and factory.
//

/// UVM run_test() implementation.
/// This is the main entry point for running UVM tests. It is called from
/// SystemVerilog code when run_test() is invoked.
///
/// @param testNameData Pointer to the test name string data
/// @param testNameLen Length of the test name string
///
/// Currently this is a stub that prints a message. Future implementation will:
/// 1. Create the test component using the UVM factory
/// 2. Execute the UVM phase sequence (build, connect, run, etc.)
/// 3. Report summarize and finish simulation
void __uvm_run_test(const char *testNameData, int64_t testNameLen);

/// UVM phase start notification.
/// This function is called at the beginning of each UVM phase.
/// It can be used to track phase execution and implement phase callbacks.
///
/// @param phaseNameData Pointer to the phase name string data
/// @param phaseNameLen Length of the phase name string
///
/// Standard UVM phases (in order):
/// - "build" - Create component hierarchy (top-down)
/// - "connect" - Connect TLM ports (bottom-up)
/// - "end_of_elaboration" - Fine-tune testbench (bottom-up)
/// - "start_of_simulation" - Get ready for simulation (bottom-up)
/// - "run" - Main test execution (task phase, time-consuming)
/// - "extract" - Extract data from DUT (bottom-up)
/// - "check" - Check DUT state (bottom-up)
/// - "report" - Report results (bottom-up)
/// - "final" - Finalize simulation (top-down)
void __uvm_phase_start(const char *phaseNameData, int64_t phaseNameLen);

/// UVM phase end notification.
/// This function is called at the end of each UVM phase.
/// It can be used to track phase execution and implement phase callbacks.
///
/// @param phaseNameData Pointer to the phase name string data
/// @param phaseNameLen Length of the phase name string
void __uvm_phase_end(const char *phaseNameData, int64_t phaseNameLen);

/// UVM phase execution.
/// Execute all standard UVM phases in sequence.
/// This is called internally by __uvm_run_test after the test component
/// is created by the factory.
///
/// The phases are executed in the standard UVM order:
/// 1. build_phase (top-down)
/// 2. connect_phase (bottom-up)
/// 3. end_of_elaboration_phase (bottom-up)
/// 4. start_of_simulation_phase (bottom-up)
/// 5. run_phase (task phase)
/// 6. extract_phase (bottom-up)
/// 7. check_phase (bottom-up)
/// 8. report_phase (bottom-up)
/// 9. final_phase (top-down)
void __uvm_execute_phases(void);

//===----------------------------------------------------------------------===//
// UVM Factory
//===----------------------------------------------------------------------===//
//
// The UVM factory provides a mechanism to create objects and components by
// name at runtime. This enables:
// - Late binding of test components (run_test("test_name"))
// - Type overrides (replace one class with another)
// - Instance overrides (replace specific instances)
//
// The factory maintains a registry of type names to creation functions.
// When create_component_by_name is called, it looks up the registered
// creator and invokes it to instantiate the component.
//

/// Callback function type for component creation.
/// This function is called by the factory to create a new component instance.
///
/// @param name The instance name for the new component
/// @param nameLen Length of the name string
/// @param parent Pointer to the parent component (NULL for root)
/// @param userData User data registered with the type
/// @return Pointer to the newly created component, or NULL on failure
typedef void *(*MooreUvmComponentCreator)(const char *name, int64_t nameLen,
                                          void *parent, void *userData);

/// Callback function type for object creation.
/// This function is called by the factory to create a new object instance.
///
/// @param name The instance name for the new object
/// @param nameLen Length of the name string
/// @param userData User data registered with the type
/// @return Pointer to the newly created object, or NULL on failure
typedef void *(*MooreUvmObjectCreator)(const char *name, int64_t nameLen,
                                       void *userData);

/// Register a component type with the factory.
/// After registration, the type can be created by name using
/// __moore_uvm_factory_create_component_by_name.
///
/// @param typeName The type name to register (e.g., "my_test")
/// @param typeNameLen Length of the type name string
/// @param creator Function to create instances of this type
/// @param userData User data to pass to the creator function
/// @return 1 on success, 0 on failure (e.g., already registered)
int32_t __moore_uvm_factory_register_component(const char *typeName,
                                               int64_t typeNameLen,
                                               MooreUvmComponentCreator creator,
                                               void *userData);

/// Register an object type with the factory.
/// After registration, the type can be created by name using
/// __moore_uvm_factory_create_object_by_name.
///
/// @param typeName The type name to register
/// @param typeNameLen Length of the type name string
/// @param creator Function to create instances of this type
/// @param userData User data to pass to the creator function
/// @return 1 on success, 0 on failure (e.g., already registered)
int32_t __moore_uvm_factory_register_object(const char *typeName,
                                            int64_t typeNameLen,
                                            MooreUvmObjectCreator creator,
                                            void *userData);

/// Create a component by type name.
/// Looks up the type in the factory registry and creates an instance.
///
/// @param typeName The registered type name
/// @param typeNameLen Length of the type name string
/// @param instName The instance name for the new component
/// @param instNameLen Length of the instance name string
/// @param parent Pointer to the parent component (NULL for root)
/// @return Pointer to the newly created component, or NULL if type not found
void *__moore_uvm_factory_create_component_by_name(const char *typeName,
                                                   int64_t typeNameLen,
                                                   const char *instName,
                                                   int64_t instNameLen,
                                                   void *parent);

/// Create an object by type name.
/// Looks up the type in the factory registry and creates an instance.
///
/// @param typeName The registered type name
/// @param typeNameLen Length of the type name string
/// @param instName The instance name for the new object
/// @param instNameLen Length of the instance name string
/// @return Pointer to the newly created object, or NULL if type not found
void *__moore_uvm_factory_create_object_by_name(const char *typeName,
                                                int64_t typeNameLen,
                                                const char *instName,
                                                int64_t instNameLen);

/// Set a type override in the factory.
/// All future requests to create the original type will instead create
/// the override type.
///
/// @param originalType The type name to override
/// @param originalTypeLen Length of the original type name
/// @param overrideType The type name to create instead
/// @param overrideTypeLen Length of the override type name
/// @param replace If true, replace existing override; if false, only add if no override exists
/// @return 1 on success, 0 on failure
int32_t __moore_uvm_factory_set_type_override(const char *originalType,
                                              int64_t originalTypeLen,
                                              const char *overrideType,
                                              int64_t overrideTypeLen,
                                              int32_t replace);

/// Check if a type is registered with the factory.
///
/// @param typeName The type name to check
/// @param typeNameLen Length of the type name string
/// @return 1 if registered, 0 if not
int32_t __moore_uvm_factory_is_type_registered(const char *typeName,
                                               int64_t typeNameLen);

/// Get the number of registered types in the factory.
/// Useful for testing and debugging.
///
/// @return The number of registered component and object types
int64_t __moore_uvm_factory_get_type_count(void);

/// Clear all registered types and overrides from the factory.
/// Primarily used for testing.
void __moore_uvm_factory_clear(void);

/// Print the factory state for debugging.
/// Lists all registered types and any active overrides.
void __moore_uvm_factory_print(void);

//===----------------------------------------------------------------------===//
// UVM Component Phase Callback Registration
//===----------------------------------------------------------------------===//
//
// These functions allow UVM components to register their phase methods with
// the runtime. When phases are executed, the runtime will call all registered
// callbacks for that phase.
//
// UVM defines 9 standard phases. Phase methods can be either:
// - Function phases (build, connect, end_of_elaboration, start_of_simulation,
//   extract, check, report, final): These are called synchronously
// - Task phases (run): These may involve time delays and are executed
//   concurrently for all components
//
// Phase execution order:
// - Top-down phases (build, final): Called from root to leaves
// - Bottom-up phases (connect, end_of_elaboration, start_of_simulation,
//   extract, check, report): Called from leaves to root
// - Task phases (run): All components execute concurrently
//

/// UVM phase enumeration.
/// These values identify the standard UVM phases.
typedef enum {
  UVM_PHASE_BUILD = 0,
  UVM_PHASE_CONNECT = 1,
  UVM_PHASE_END_OF_ELABORATION = 2,
  UVM_PHASE_START_OF_SIMULATION = 3,
  UVM_PHASE_RUN = 4,
  UVM_PHASE_EXTRACT = 5,
  UVM_PHASE_CHECK = 6,
  UVM_PHASE_REPORT = 7,
  UVM_PHASE_FINAL = 8,
  UVM_PHASE_COUNT = 9
} MooreUvmPhase;

/// Phase callback function type for function phases.
/// This is used for all phases except run_phase (which is a task).
///
/// @param component Pointer to the component instance
/// @param phase The phase being executed (for passing to super.XXX_phase())
/// @param userData User data registered with the callback
typedef void (*MooreUvmPhaseCallback)(void *component, void *phase,
                                      void *userData);

/// Phase callback function type for task phases (run_phase).
/// Task phases may involve time and run concurrently.
/// In the current implementation, these are called synchronously but
/// the signature allows for future async implementation.
///
/// @param component Pointer to the component instance
/// @param phase The phase being executed
/// @param userData User data registered with the callback
typedef void (*MooreUvmTaskPhaseCallback)(void *component, void *phase,
                                          void *userData);

/// Register a UVM component with the phase system.
/// The component will be called during phase execution based on its position
/// in the component hierarchy.
///
/// @param component Pointer to the component instance
/// @param name The component's instance name
/// @param nameLen Length of the name string
/// @param parent Pointer to the parent component (NULL for root)
/// @param depth Hierarchy depth (0 for root, 1 for direct children, etc.)
/// @return A handle for the registered component (0 on failure)
int64_t __moore_uvm_register_component(void *component, const char *name,
                                       int64_t nameLen, void *parent,
                                       int32_t depth);

/// Unregister a UVM component from the phase system.
/// This should be called when a component is destroyed.
///
/// @param handle The handle returned by __moore_uvm_register_component
void __moore_uvm_unregister_component(int64_t handle);

/// Register a phase callback for a component.
/// The callback will be invoked when the specified phase is executed.
///
/// @param handle The component handle from __moore_uvm_register_component
/// @param phase The phase to register for
/// @param callback The callback function to invoke
/// @param userData User data to pass to the callback
void __moore_uvm_set_phase_callback(int64_t handle, MooreUvmPhase phase,
                                    MooreUvmPhaseCallback callback,
                                    void *userData);

/// Register a task phase callback for a component (for run_phase).
/// The callback will be invoked when the run phase is executed.
///
/// @param handle The component handle from __moore_uvm_register_component
/// @param callback The callback function to invoke
/// @param userData User data to pass to the callback
void __moore_uvm_set_run_phase_callback(int64_t handle,
                                        MooreUvmTaskPhaseCallback callback,
                                        void *userData);

/// Get the number of registered components.
/// Useful for testing and debugging.
///
/// @return The number of components currently registered
int64_t __moore_uvm_get_component_count(void);

/// Clear all registered components and callbacks.
/// This resets the phase system to its initial state.
/// Primarily used for testing.
void __moore_uvm_clear_components(void);

/// Set a global phase start callback.
/// This callback is invoked before any component callbacks for the phase.
///
/// @param callback The callback function (NULL to disable)
/// @param userData User data to pass to the callback
void __moore_uvm_set_global_phase_start_callback(
    void (*callback)(MooreUvmPhase phase, const char *phaseName, void *userData),
    void *userData);

/// Set a global phase end callback.
/// This callback is invoked after all component callbacks for the phase.
///
/// @param callback The callback function (NULL to disable)
/// @param userData User data to pass to the callback
void __moore_uvm_set_global_phase_end_callback(
    void (*callback)(MooreUvmPhase phase, const char *phaseName, void *userData),
    void *userData);

//===----------------------------------------------------------------------===//
// TLM Port/Export Runtime Infrastructure
//===----------------------------------------------------------------------===//
//
// These functions implement the UVM TLM (Transaction Level Modeling) port/export
// infrastructure needed for AVIP monitor -> scoreboard communication.
//
// The TLM infrastructure supports:
// - Analysis ports (1-to-N broadcast communication)
// - Analysis FIFOs (queue-based transaction storage)
// - Port-to-export connections
// - Blocking get operations on FIFOs
//
// Key usage patterns:
// 1. Monitor -> Scoreboard via Analysis FIFO
//    - Monitor writes transactions via analysis_port.write()
//    - Scoreboard gets transactions via fifo.get()
// 2. Monitor -> Coverage via uvm_subscriber
//    - Subscriber receives transactions via write() callback
//
//===----------------------------------------------------------------------===//

/// Handle type for TLM ports.
typedef int64_t MooreTlmPortHandle;

/// Handle type for TLM FIFOs.
typedef int64_t MooreTlmFifoHandle;

/// Invalid handle value.
#define MOORE_TLM_INVALID_HANDLE (-1)

/// TLM port types.
typedef enum {
  MOORE_TLM_PORT_ANALYSIS = 0,    // uvm_analysis_port
  MOORE_TLM_PORT_BLOCKING_GET = 1, // uvm_blocking_get_port
  MOORE_TLM_PORT_BLOCKING_PUT = 2, // uvm_blocking_put_port
  MOORE_TLM_PORT_SEQ_ITEM = 3      // uvm_seq_item_pull_port
} MooreTlmPortType;

/// Write callback function type for subscribers.
/// Called when a transaction is written to an analysis port.
/// @param subscriber User data pointer for the subscriber
/// @param transaction Pointer to the transaction data
/// @param transactionSize Size of the transaction in bytes
typedef void (*MooreTlmWriteCallback)(void *subscriber, void *transaction,
                                      int64_t transactionSize);

//===----------------------------------------------------------------------===//
// TLM Port Operations
//===----------------------------------------------------------------------===//

/// Create a new TLM port.
/// @param name Port name (for debugging/tracing)
/// @param nameLen Length of the name string
/// @param parent Parent component handle (from __moore_uvm_register_component)
/// @param portType Type of the port (MOORE_TLM_PORT_*)
/// @return Handle to the created port, or MOORE_TLM_INVALID_HANDLE on failure
MooreTlmPortHandle __moore_tlm_port_create(const char *name, int64_t nameLen,
                                           int64_t parent,
                                           MooreTlmPortType portType);

/// Destroy a TLM port.
/// @param port Handle to the port to destroy
void __moore_tlm_port_destroy(MooreTlmPortHandle port);

/// Connect a TLM port to an export (or another port).
/// This implements the connect() method from UVM TLM ports.
///
/// @param port Handle to the initiating port
/// @param export_ Handle to the target export/port
/// @return 1 on success, 0 on failure
int32_t __moore_tlm_port_connect(MooreTlmPortHandle port,
                                 MooreTlmPortHandle export_);

/// Write (broadcast) a transaction to all connected subscribers.
/// This implements the write() method of uvm_analysis_port.
///
/// For analysis ports, this broadcasts the transaction to ALL connected
/// exports/imps (1-to-N communication).
///
/// @param port Handle to the analysis port
/// @param transaction Pointer to the transaction data
/// @param transactionSize Size of the transaction in bytes
void __moore_tlm_port_write(MooreTlmPortHandle port, void *transaction,
                            int64_t transactionSize);

/// Get the name of a TLM port.
/// @param port Handle to the port
/// @return The port name as a MooreString
MooreString __moore_tlm_port_get_name(MooreTlmPortHandle port);

/// Get the number of connections on a port.
/// @param port Handle to the port
/// @return Number of connected exports/subscribers
int64_t __moore_tlm_port_get_num_connections(MooreTlmPortHandle port);

//===----------------------------------------------------------------------===//
// TLM FIFO Operations
//===----------------------------------------------------------------------===//

/// Create a new TLM FIFO.
/// This implements uvm_tlm_fifo and uvm_tlm_analysis_fifo.
///
/// @param name FIFO name (for debugging)
/// @param nameLen Length of the name string
/// @param parent Parent component handle
/// @param maxSize Maximum FIFO size (0 for unbounded, which is typical for
///                analysis FIFOs)
/// @param elementSize Size of each element in bytes
/// @return Handle to the created FIFO, or MOORE_TLM_INVALID_HANDLE on failure
MooreTlmFifoHandle __moore_tlm_fifo_create(const char *name, int64_t nameLen,
                                           int64_t parent, int64_t maxSize,
                                           int64_t elementSize);

/// Destroy a TLM FIFO.
/// @param fifo Handle to the FIFO to destroy
void __moore_tlm_fifo_destroy(MooreTlmFifoHandle fifo);

/// Get the analysis export handle for a TLM analysis FIFO.
/// This returns the handle that should be connected to analysis ports.
///
/// @param fifo Handle to the FIFO
/// @return Handle to the analysis_export port
MooreTlmPortHandle __moore_tlm_fifo_get_analysis_export(MooreTlmFifoHandle fifo);

/// Put a transaction into the FIFO (non-blocking).
/// @param fifo Handle to the FIFO
/// @param transaction Pointer to the transaction data
/// @param transactionSize Size of the transaction in bytes
/// @return 1 if successful, 0 if FIFO is full (bounded FIFO only)
int32_t __moore_tlm_fifo_try_put(MooreTlmFifoHandle fifo, void *transaction,
                                 int64_t transactionSize);

/// Put a transaction into the FIFO (blocking).
/// This blocks until space is available (for bounded FIFOs).
///
/// @param fifo Handle to the FIFO
/// @param transaction Pointer to the transaction data
/// @param transactionSize Size of the transaction in bytes
void __moore_tlm_fifo_put(MooreTlmFifoHandle fifo, void *transaction,
                          int64_t transactionSize);

/// Get a transaction from the FIFO (blocking).
/// This implements the blocking get() method of uvm_tlm_fifo.
/// The call blocks until a transaction is available.
///
/// @param fifo Handle to the FIFO
/// @param transaction Pointer where the transaction data will be copied
/// @param transactionSize Size of the transaction buffer in bytes
/// @return 1 if a transaction was retrieved, 0 on error
int32_t __moore_tlm_fifo_get(MooreTlmFifoHandle fifo, void *transaction,
                             int64_t transactionSize);

/// Try to get a transaction from the FIFO (non-blocking).
/// @param fifo Handle to the FIFO
/// @param transaction Pointer where the transaction data will be copied
/// @param transactionSize Size of the transaction buffer in bytes
/// @return 1 if a transaction was retrieved, 0 if FIFO is empty
int32_t __moore_tlm_fifo_try_get(MooreTlmFifoHandle fifo, void *transaction,
                                 int64_t transactionSize);

/// Peek at the front transaction without removing it (blocking).
/// @param fifo Handle to the FIFO
/// @param transaction Pointer where the transaction data will be copied
/// @param transactionSize Size of the transaction buffer in bytes
/// @return 1 if a transaction was peeked, 0 on error
int32_t __moore_tlm_fifo_peek(MooreTlmFifoHandle fifo, void *transaction,
                              int64_t transactionSize);

/// Try to peek at the front transaction (non-blocking).
/// @param fifo Handle to the FIFO
/// @param transaction Pointer where the transaction data will be copied
/// @param transactionSize Size of the transaction buffer in bytes
/// @return 1 if a transaction was peeked, 0 if FIFO is empty
int32_t __moore_tlm_fifo_try_peek(MooreTlmFifoHandle fifo, void *transaction,
                                  int64_t transactionSize);

/// Get the current number of items in the FIFO.
/// @param fifo Handle to the FIFO
/// @return Number of items currently in the FIFO
int64_t __moore_tlm_fifo_size(MooreTlmFifoHandle fifo);

/// Check if the FIFO is empty.
/// @param fifo Handle to the FIFO
/// @return 1 if empty, 0 otherwise
int32_t __moore_tlm_fifo_is_empty(MooreTlmFifoHandle fifo);

/// Check if the FIFO is full (only relevant for bounded FIFOs).
/// @param fifo Handle to the FIFO
/// @return 1 if full, 0 otherwise (always 0 for unbounded FIFOs)
int32_t __moore_tlm_fifo_is_full(MooreTlmFifoHandle fifo);

/// Flush (clear) all items from the FIFO.
/// @param fifo Handle to the FIFO
void __moore_tlm_fifo_flush(MooreTlmFifoHandle fifo);

/// Check if put can proceed without blocking.
/// This implements the can_put() method of uvm_tlm_fifo.
/// For unbounded FIFOs, always returns 1.
/// For bounded FIFOs, returns 1 if there is free space.
/// @param fifo Handle to the FIFO
/// @return 1 if put can proceed, 0 if FIFO is full
int32_t __moore_tlm_fifo_can_put(MooreTlmFifoHandle fifo);

/// Check if get can proceed without blocking.
/// This implements the can_get() method of uvm_tlm_fifo.
/// @param fifo Handle to the FIFO
/// @return 1 if get can proceed, 0 if FIFO is empty
int32_t __moore_tlm_fifo_can_get(MooreTlmFifoHandle fifo);

/// Get the number of items currently in the FIFO.
/// This implements the used() method of uvm_tlm_fifo.
/// Alias for __moore_tlm_fifo_size().
/// @param fifo Handle to the FIFO
/// @return Number of items in the FIFO
int64_t __moore_tlm_fifo_used(MooreTlmFifoHandle fifo);

/// Get the number of free slots in the FIFO.
/// This implements the free() method of uvm_tlm_fifo.
/// For unbounded FIFOs, returns INT64_MAX.
/// For bounded FIFOs, returns (maxSize - currentSize).
/// @param fifo Handle to the FIFO
/// @return Number of free slots
int64_t __moore_tlm_fifo_free(MooreTlmFifoHandle fifo);

/// Get the maximum capacity of the FIFO.
/// This implements the capacity() method of uvm_tlm_fifo.
/// @param fifo Handle to the FIFO
/// @return Maximum capacity (0 for unbounded FIFO)
int64_t __moore_tlm_fifo_capacity(MooreTlmFifoHandle fifo);

//===----------------------------------------------------------------------===//
// TLM Subscriber Operations
//===----------------------------------------------------------------------===//

/// Register a write callback for a subscriber.
/// This is used to implement uvm_subscriber pattern where the subscriber's
/// write() method is called when a transaction is received.
///
/// @param port Handle to the analysis imp/export
/// @param callback The callback function to invoke
/// @param userData User data to pass to the callback (typically 'this' pointer)
void __moore_tlm_subscriber_set_write_callback(MooreTlmPortHandle port,
                                               MooreTlmWriteCallback callback,
                                               void *userData);

//===----------------------------------------------------------------------===//
// TLM Debugging/Tracing
//===----------------------------------------------------------------------===//

/// Enable or disable TLM tracing.
/// When enabled, all TLM operations are logged for debugging.
/// @param enable 1 to enable, 0 to disable
void __moore_tlm_set_trace_enabled(int32_t enable);

/// Check if TLM tracing is enabled.
/// @return 1 if enabled, 0 otherwise
int32_t __moore_tlm_is_trace_enabled(void);

/// Print TLM connection topology for debugging.
void __moore_tlm_print_topology(void);

/// Get statistics about TLM operations.
/// @param totalConnections Output: total number of connections made
/// @param totalWrites Output: total number of write() calls
/// @param totalGets Output: total number of get() calls
void __moore_tlm_get_statistics(int64_t *totalConnections, int64_t *totalWrites,
                                int64_t *totalGets);

//===----------------------------------------------------------------------===//
// UVM Objection System
//===----------------------------------------------------------------------===//
//
// The UVM objection system controls phase transitions in testbenches.
// Components raise objections to prevent a phase from ending, and drop them
// when their work is complete. The phase controller waits until all objections
// are dropped (and an optional drain time has elapsed) before advancing.
//
// Key concepts:
// - Objection pool: A named collection of objections for a specific phase
// - Raise/drop: Increment/decrement the objection count (with optional context)
// - Drain time: Time to wait after count reaches zero before ending phase
// - Hierarchical context: Objections can be associated with component paths
//
//===----------------------------------------------------------------------===//

/// Handle type for objection pools.
typedef int64_t MooreObjectionHandle;

/// Invalid objection handle value.
#define MOORE_OBJECTION_INVALID_HANDLE (-1)

/// Create a new objection pool for a phase.
/// Each phase should have its own objection pool to track when the phase
/// can complete.
/// @param phaseName Name of the phase (e.g., "run", "main", "shutdown")
/// @param phaseNameLen Length of the phase name string
/// @return Handle to the created objection pool, or MOORE_OBJECTION_INVALID_HANDLE on failure
MooreObjectionHandle __moore_objection_create(const char *phaseName,
                                               int64_t phaseNameLen);

/// Destroy an objection pool.
/// @param objection Handle to the objection pool to destroy
void __moore_objection_destroy(MooreObjectionHandle objection);

/// Raise an objection to prevent a phase from ending.
/// This increments the objection count. The phase will not end until all
/// raised objections are dropped.
/// @param objection Handle to the objection pool
/// @param context Component path or context string (may be NULL)
/// @param contextLen Length of the context string (0 if NULL)
/// @param description Optional description of why objection is raised (may be NULL)
/// @param descriptionLen Length of the description (0 if NULL)
/// @param count Number of objections to raise (typically 1)
void __moore_objection_raise(MooreObjectionHandle objection,
                             const char *context, int64_t contextLen,
                             const char *description, int64_t descriptionLen,
                             int64_t count);

/// Drop an objection, potentially allowing the phase to end.
/// This decrements the objection count. When all objections are dropped
/// (count reaches zero), the phase may end after the drain time.
/// @param objection Handle to the objection pool
/// @param context Component path or context string (may be NULL)
/// @param contextLen Length of the context string (0 if NULL)
/// @param description Optional description of why objection is dropped (may be NULL)
/// @param descriptionLen Length of the description (0 if NULL)
/// @param count Number of objections to drop (typically 1)
void __moore_objection_drop(MooreObjectionHandle objection,
                            const char *context, int64_t contextLen,
                            const char *description, int64_t descriptionLen,
                            int64_t count);

/// Get the total objection count for a phase.
/// @param objection Handle to the objection pool
/// @return Current total count of raised objections
int64_t __moore_objection_get_count(MooreObjectionHandle objection);

/// Get the objection count for a specific context.
/// @param objection Handle to the objection pool
/// @param context Component path or context string
/// @param contextLen Length of the context string
/// @return Count of objections raised by the specified context
int64_t __moore_objection_get_count_by_context(MooreObjectionHandle objection,
                                                const char *context,
                                                int64_t contextLen);

/// Set the drain time for an objection pool.
/// The drain time is the amount of time to wait after all objections are
/// dropped before signaling that the phase can end. This allows for any
/// final cleanup or propagation delays.
/// @param objection Handle to the objection pool
/// @param drainTime Drain time in simulation time units (0 for immediate)
void __moore_objection_set_drain_time(MooreObjectionHandle objection,
                                       int64_t drainTime);

/// Get the drain time for an objection pool.
/// @param objection Handle to the objection pool
/// @return Current drain time setting
int64_t __moore_objection_get_drain_time(MooreObjectionHandle objection);

/// Wait until all objections are dropped and drain time has elapsed.
/// This is a blocking call used by the phase controller to wait for
/// the phase to complete. In a simulation environment, this would
/// integrate with the simulation scheduler.
/// @param objection Handle to the objection pool
/// @return 1 when complete (all objections dropped + drain time elapsed), 0 on error
int32_t __moore_objection_wait_for_zero(MooreObjectionHandle objection);

/// Check if all objections have been dropped (non-blocking).
/// @param objection Handle to the objection pool
/// @return 1 if count is zero, 0 if objections are still raised
int32_t __moore_objection_is_zero(MooreObjectionHandle objection);

/// Get the name of the phase associated with an objection pool.
/// @param objection Handle to the objection pool
/// @return The phase name as a MooreString
MooreString __moore_objection_get_phase_name(MooreObjectionHandle objection);

/// Enable or disable objection tracing for debugging.
/// When enabled, all objection operations are logged.
/// @param enable 1 to enable, 0 to disable
void __moore_objection_set_trace_enabled(int32_t enable);

/// Check if objection tracing is enabled.
/// @return 1 if enabled, 0 otherwise
int32_t __moore_objection_is_trace_enabled(void);

/// Print a summary of all objection pools and their current state.
/// Useful for debugging phase hang issues.
void __moore_objection_print_summary(void);

//===----------------------------------------------------------------------===//
// UVM Sequence/Sequencer Infrastructure
//===----------------------------------------------------------------------===//
//
// The UVM sequence/sequencer mechanism provides a layered stimulus generation
// framework. Sequences generate transactions (sequence items) that are sent
// to drivers through sequencers.
//
// Key concepts:
// - Sequencer: Manages sequence execution and arbitration between sequences
// - Sequence: Generates a stream of transactions (sequence items)
// - start_item/finish_item: Handshake protocol between sequence and driver
// - Arbitration: Determines which sequence gets access to the driver
//
// Flow:
// 1. Sequence calls start_item() to request driver access
// 2. Sequencer arbitrates if multiple sequences are waiting
// 3. Driver calls get_next_item() to receive the item
// 4. Driver processes the item
// 5. Driver calls item_done() to signal completion
// 6. Sequence's finish_item() returns
//
//===----------------------------------------------------------------------===//

/// Handle type for sequencers.
typedef int64_t MooreSequencerHandle;

/// Handle type for sequences.
typedef int64_t MooreSequenceHandle;

/// Invalid handle values.
#define MOORE_SEQUENCER_INVALID_HANDLE (-1)
#define MOORE_SEQUENCE_INVALID_HANDLE (-1)

/// Sequencer arbitration modes.
/// These determine how the sequencer selects between competing sequences.
typedef enum {
  /// First-in, first-out - sequences get access in the order they requested
  MOORE_SEQ_ARB_FIFO = 0,
  /// Random selection among waiting sequences
  MOORE_SEQ_ARB_RANDOM = 1,
  /// Weighted random based on sequence priority
  MOORE_SEQ_ARB_WEIGHTED = 2,
  /// Strict priority - highest priority sequence always wins
  MOORE_SEQ_ARB_STRICT_FIFO = 3,
  /// Strict random among highest priority sequences
  MOORE_SEQ_ARB_STRICT_RANDOM = 4,
  /// User-defined arbitration (via callback)
  MOORE_SEQ_ARB_USER = 5
} MooreSeqArbMode;

/// Sequence state.
typedef enum {
  MOORE_SEQ_STATE_IDLE = 0,        ///< Sequence not started
  MOORE_SEQ_STATE_RUNNING = 1,     ///< Sequence body is executing
  MOORE_SEQ_STATE_WAITING = 2,     ///< Waiting for driver (start_item)
  MOORE_SEQ_STATE_FINISHED = 3,    ///< Sequence completed
  MOORE_SEQ_STATE_STOPPED = 4      ///< Sequence stopped externally
} MooreSeqState;

/// Callback type for sequence body execution.
/// The sequence body generates transactions by calling start_item/finish_item.
/// @param sequence Handle to the sequence being executed
/// @param userData User-provided context data
typedef void (*MooreSequenceBodyCallback)(MooreSequenceHandle sequence,
                                          void *userData);

/// Callback type for user-defined arbitration.
/// @param sequencer Handle to the sequencer
/// @param waitingSequences Array of sequence handles waiting for access
/// @param numWaiting Number of sequences in the array
/// @param userData User-provided context data
/// @return Index of the selected sequence (0 to numWaiting-1)
typedef int32_t (*MooreSeqArbCallback)(MooreSequencerHandle sequencer,
                                       MooreSequenceHandle *waitingSequences,
                                       int32_t numWaiting, void *userData);

//===----------------------------------------------------------------------===//
// Sequencer Operations
//===----------------------------------------------------------------------===//

/// Create a new sequencer.
/// @param name Sequencer name (for debugging/tracing)
/// @param nameLen Length of the name string
/// @param parent Parent component handle (0 for top-level)
/// @return Handle to the created sequencer, or MOORE_SEQUENCER_INVALID_HANDLE
MooreSequencerHandle __moore_sequencer_create(const char *name, int64_t nameLen,
                                               int64_t parent);

/// Destroy a sequencer.
/// Stops all running sequences and releases resources.
/// @param sequencer Handle to the sequencer to destroy
void __moore_sequencer_destroy(MooreSequencerHandle sequencer);

/// Start the sequencer, enabling it to process sequences.
/// @param sequencer Handle to the sequencer
void __moore_sequencer_start(MooreSequencerHandle sequencer);

/// Stop the sequencer, preventing new sequences from starting.
/// Running sequences may complete or be forcibly stopped.
/// @param sequencer Handle to the sequencer
void __moore_sequencer_stop(MooreSequencerHandle sequencer);

/// Check if the sequencer is running.
/// @param sequencer Handle to the sequencer
/// @return 1 if running, 0 otherwise
int32_t __moore_sequencer_is_running(MooreSequencerHandle sequencer);

/// Set the arbitration mode for a sequencer.
/// @param sequencer Handle to the sequencer
/// @param mode Arbitration mode (MOORE_SEQ_ARB_*)
void __moore_sequencer_set_arbitration(MooreSequencerHandle sequencer,
                                        MooreSeqArbMode mode);

/// Get the current arbitration mode.
/// @param sequencer Handle to the sequencer
/// @return Current arbitration mode
MooreSeqArbMode __moore_sequencer_get_arbitration(MooreSequencerHandle sequencer);

/// Set user-defined arbitration callback.
/// Only used when arbitration mode is MOORE_SEQ_ARB_USER.
/// @param sequencer Handle to the sequencer
/// @param callback Arbitration callback function
/// @param userData User data to pass to the callback
void __moore_sequencer_set_arb_callback(MooreSequencerHandle sequencer,
                                         MooreSeqArbCallback callback,
                                         void *userData);

/// Get the name of a sequencer.
/// @param sequencer Handle to the sequencer
/// @return The sequencer name as a MooreString
MooreString __moore_sequencer_get_name(MooreSequencerHandle sequencer);

/// Get the number of sequences currently waiting for access.
/// @param sequencer Handle to the sequencer
/// @return Number of waiting sequences
int32_t __moore_sequencer_get_num_waiting(MooreSequencerHandle sequencer);

//===----------------------------------------------------------------------===//
// Sequence Operations
//===----------------------------------------------------------------------===//

/// Create a new sequence.
/// @param name Sequence name (for debugging/tracing)
/// @param nameLen Length of the name string
/// @param priority Sequence priority (higher = more likely to be selected)
/// @return Handle to the created sequence, or MOORE_SEQUENCE_INVALID_HANDLE
MooreSequenceHandle __moore_sequence_create(const char *name, int64_t nameLen,
                                             int32_t priority);

/// Destroy a sequence.
/// The sequence must be stopped or finished before destruction.
/// @param sequence Handle to the sequence to destroy
void __moore_sequence_destroy(MooreSequenceHandle sequence);

/// Start a sequence on a sequencer.
/// This begins execution of the sequence body on the specified sequencer.
/// The call blocks until the sequence completes (in synchronous mode).
/// @param sequence Handle to the sequence
/// @param sequencer Handle to the target sequencer
/// @param body Callback function containing the sequence body
/// @param userData User data to pass to the body callback
/// @return 1 on successful completion, 0 on failure or if stopped
int32_t __moore_sequence_start(MooreSequenceHandle sequence,
                                MooreSequencerHandle sequencer,
                                MooreSequenceBodyCallback body,
                                void *userData);

/// Start a sequence asynchronously (non-blocking).
/// The sequence runs in the background. Use __moore_sequence_wait() to wait.
/// @param sequence Handle to the sequence
/// @param sequencer Handle to the target sequencer
/// @param body Callback function containing the sequence body
/// @param userData User data to pass to the body callback
/// @return 1 if started successfully, 0 on failure
int32_t __moore_sequence_start_async(MooreSequenceHandle sequence,
                                      MooreSequencerHandle sequencer,
                                      MooreSequenceBodyCallback body,
                                      void *userData);

/// Wait for an async sequence to complete.
/// @param sequence Handle to the sequence
/// @return 1 on successful completion, 0 on failure
int32_t __moore_sequence_wait(MooreSequenceHandle sequence);

/// Stop a running sequence.
/// @param sequence Handle to the sequence
void __moore_sequence_stop(MooreSequenceHandle sequence);

/// Get the current state of a sequence.
/// @param sequence Handle to the sequence
/// @return Current sequence state (MOORE_SEQ_STATE_*)
MooreSeqState __moore_sequence_get_state(MooreSequenceHandle sequence);

/// Get the name of a sequence.
/// @param sequence Handle to the sequence
/// @return The sequence name as a MooreString
MooreString __moore_sequence_get_name(MooreSequenceHandle sequence);

/// Get the priority of a sequence.
/// @param sequence Handle to the sequence
/// @return Sequence priority
int32_t __moore_sequence_get_priority(MooreSequenceHandle sequence);

/// Set the priority of a sequence.
/// @param sequence Handle to the sequence
/// @param priority New priority value
void __moore_sequence_set_priority(MooreSequenceHandle sequence,
                                    int32_t priority);

//===----------------------------------------------------------------------===//
// Sequence-Driver Handshake (start_item/finish_item)
//===----------------------------------------------------------------------===//

/// Begin a sequence item transfer (sequence side).
/// Called by the sequence to request access to the driver.
/// This function blocks until the driver is ready to receive the item.
/// @param sequence Handle to the sequence
/// @param item Pointer to the sequence item (transaction)
/// @param itemSize Size of the sequence item in bytes
/// @return 1 if access granted, 0 on failure (e.g., sequence stopped)
int32_t __moore_sequence_start_item(MooreSequenceHandle sequence,
                                     void *item, int64_t itemSize);

/// Complete a sequence item transfer (sequence side).
/// Called by the sequence after the item has been prepared.
/// This function blocks until the driver signals item_done().
/// @param sequence Handle to the sequence
/// @param item Pointer to the sequence item (may have been modified)
/// @param itemSize Size of the sequence item in bytes
/// @return 1 on success, 0 on failure
int32_t __moore_sequence_finish_item(MooreSequenceHandle sequence,
                                      void *item, int64_t itemSize);

/// Get the next item from a sequencer (driver side).
/// Called by the driver to receive a transaction from a sequence.
/// This function blocks until an item is available.
/// @param sequencer Handle to the sequencer
/// @param item Pointer where the item will be copied
/// @param itemSize Size of the item buffer in bytes
/// @return 1 if item received, 0 on failure
int32_t __moore_sequencer_get_next_item(MooreSequencerHandle sequencer,
                                         void *item, int64_t itemSize);

/// Try to get the next item without blocking (driver side).
/// @param sequencer Handle to the sequencer
/// @param item Pointer where the item will be copied
/// @param itemSize Size of the item buffer in bytes
/// @return 1 if item received, 0 if no item available
int32_t __moore_sequencer_try_get_next_item(MooreSequencerHandle sequencer,
                                             void *item, int64_t itemSize);

/// Signal that the current item has been processed (driver side).
/// This unblocks the sequence's finish_item() call.
/// @param sequencer Handle to the sequencer
void __moore_sequencer_item_done(MooreSequencerHandle sequencer);

/// Signal item done with a response (driver side).
/// Used when the driver needs to send response data back to the sequence.
/// @param sequencer Handle to the sequencer
/// @param response Pointer to the response data
/// @param responseSize Size of the response in bytes
void __moore_sequencer_item_done_with_response(MooreSequencerHandle sequencer,
                                                void *response,
                                                int64_t responseSize);

/// Peek at the next item without removing it (driver side).
/// @param sequencer Handle to the sequencer
/// @param item Pointer where the item will be copied
/// @param itemSize Size of the item buffer in bytes
/// @return 1 if item available, 0 otherwise
int32_t __moore_sequencer_peek_next_item(MooreSequencerHandle sequencer,
                                          void *item, int64_t itemSize);

/// Check if items are available in the sequencer.
/// @param sequencer Handle to the sequencer
/// @return 1 if items are available, 0 otherwise
int32_t __moore_sequencer_has_items(MooreSequencerHandle sequencer);

//===----------------------------------------------------------------------===//
// Sequence/Sequencer Debugging
//===----------------------------------------------------------------------===//

/// Enable or disable sequence/sequencer tracing.
/// When enabled, all sequence operations are logged for debugging.
/// @param enable 1 to enable, 0 to disable
void __moore_seq_set_trace_enabled(int32_t enable);

/// Check if sequence tracing is enabled.
/// @return 1 if enabled, 0 otherwise
int32_t __moore_seq_is_trace_enabled(void);

/// Print a summary of all sequencers and their state.
void __moore_seq_print_summary(void);

/// Get statistics about sequence operations.
/// @param totalSequences Output: total number of sequences created
/// @param totalItems Output: total number of items transferred
/// @param totalArbitrations Output: total number of arbitration decisions
void __moore_seq_get_statistics(int64_t *totalSequences, int64_t *totalItems,
                                 int64_t *totalArbitrations);

//===----------------------------------------------------------------------===//
// SystemVerilog Semaphore Support
//===----------------------------------------------------------------------===//
//
// Semaphores are synchronization primitives in SystemVerilog used for
// controlling access to shared resources. They are commonly used in UVM drivers
// to coordinate multiple threads accessing shared bus channels.
//
// In the AVIPs, semaphores are heavily used in complex drivers like AXI4 to
// coordinate write data, write response, and read data channel threads:
//
//   semaphore write_data_channel_key;
//   write_data_channel_key = new(1);  // One key initially
//   write_data_channel_key.get(1);    // Block until key available
//   // ... do work ...
//   write_data_channel_key.put(1);    // Release key
//
// Key methods:
// - new(int keyCount): Create semaphore with initial keys
// - put(int keyCount): Add keys to semaphore
// - get(int keyCount): Block and acquire keys (blocking)
// - try_get(int keyCount): Try to acquire keys (non-blocking)
//
//===----------------------------------------------------------------------===//

/// Handle type for semaphores.
typedef int64_t MooreSemaphoreHandle;

/// Invalid semaphore handle value.
#define MOORE_SEMAPHORE_INVALID_HANDLE (-1)

/// Create a new semaphore with an initial key count.
/// @param keyCount Initial number of keys in the semaphore
/// @return Handle to the created semaphore, or MOORE_SEMAPHORE_INVALID_HANDLE
MooreSemaphoreHandle __moore_semaphore_create(int32_t keyCount);

/// Destroy a semaphore and free its resources.
/// @param sem Handle to the semaphore to destroy
void __moore_semaphore_destroy(MooreSemaphoreHandle sem);

/// Put (release) keys into the semaphore.
/// This increments the key count and may unblock waiting threads.
/// @param sem Handle to the semaphore
/// @param keyCount Number of keys to add (must be positive)
void __moore_semaphore_put(MooreSemaphoreHandle sem, int32_t keyCount);

/// Get (acquire) keys from the semaphore (blocking).
/// This blocks until the requested number of keys are available.
/// The key count is decremented atomically once acquired.
/// @param sem Handle to the semaphore
/// @param keyCount Number of keys to acquire (must be positive)
void __moore_semaphore_get(MooreSemaphoreHandle sem, int32_t keyCount);

/// Try to get keys from the semaphore (non-blocking).
/// If the requested number of keys are available, acquires them and returns 1.
/// Otherwise, returns 0 immediately without blocking.
/// @param sem Handle to the semaphore
/// @param keyCount Number of keys to try to acquire (must be positive)
/// @return 1 if keys were acquired, 0 if not enough keys available
int32_t __moore_semaphore_try_get(MooreSemaphoreHandle sem, int32_t keyCount);

/// Get the current key count in the semaphore.
/// Note: This is mainly for debugging; the value may change immediately
/// after reading if other threads are using the semaphore.
/// @param sem Handle to the semaphore
/// @return Current number of available keys
int32_t __moore_semaphore_get_key_count(MooreSemaphoreHandle sem);

//===----------------------------------------------------------------------===//
// UVM Scoreboard Utilities
//===----------------------------------------------------------------------===//
//
// The UVM scoreboard pattern compares expected vs actual transactions to verify
// design correctness. Scoreboards typically receive transactions via TLM analysis
// FIFOs from monitors and compare them against expected results.
//
// Key concepts:
// - Expected vs Actual: Two FIFOs hold transactions from reference model and DUT
// - Transaction matching: Comparison function determines if transactions match
// - Statistics: Track matches, mismatches, and pending transactions
// - Reporting: Generate verification summary at end of simulation
//
// Integration with TLM:
// - Scoreboards use analysis FIFOs to receive transactions
// - Monitor writes to analysis port -> FIFO -> scoreboard reads
// - Supports in-order comparison (FIFO) or out-of-order matching
//
//===----------------------------------------------------------------------===//

/// Handle type for scoreboards.
typedef int64_t MooreScoreboardHandle;

/// Invalid scoreboard handle value.
#define MOORE_SCOREBOARD_INVALID_HANDLE (-1)

/// Comparison result values.
typedef enum {
  MOORE_SCOREBOARD_MATCH = 0,      ///< Transactions match
  MOORE_SCOREBOARD_MISMATCH = 1,   ///< Transactions do not match
  MOORE_SCOREBOARD_TIMEOUT = 2     ///< Comparison timed out (no transaction available)
} MooreScoreboardCompareResult;

/// Transaction compare callback function type.
/// Called to compare an expected transaction with an actual transaction.
/// @param expected Pointer to the expected transaction data
/// @param actual Pointer to the actual transaction data
/// @param transactionSize Size of the transaction in bytes
/// @param userData User-provided context data
/// @return 1 if transactions match, 0 if they do not match
typedef int32_t (*MooreScoreboardCompareCallback)(const void *expected,
                                                   const void *actual,
                                                   int64_t transactionSize,
                                                   void *userData);

/// Mismatch callback function type.
/// Called when a mismatch is detected between expected and actual transactions.
/// @param expected Pointer to the expected transaction data
/// @param actual Pointer to the actual transaction data
/// @param transactionSize Size of the transaction in bytes
/// @param userData User-provided context data
typedef void (*MooreScoreboardMismatchCallback)(const void *expected,
                                                 const void *actual,
                                                 int64_t transactionSize,
                                                 void *userData);

//===----------------------------------------------------------------------===//
// Scoreboard Creation and Configuration
//===----------------------------------------------------------------------===//

/// Create a new scoreboard.
/// @param name Scoreboard name (for debugging/reporting)
/// @param nameLen Length of the name string
/// @param transactionSize Size of each transaction in bytes
/// @return Handle to the created scoreboard, or MOORE_SCOREBOARD_INVALID_HANDLE
MooreScoreboardHandle __moore_scoreboard_create(const char *name,
                                                 int64_t nameLen,
                                                 int64_t transactionSize);

/// Destroy a scoreboard and release its resources.
/// @param scoreboard Handle to the scoreboard to destroy
void __moore_scoreboard_destroy(MooreScoreboardHandle scoreboard);

/// Set the comparison callback for a scoreboard.
/// If not set, a default byte-by-byte comparison is used.
/// @param scoreboard Handle to the scoreboard
/// @param callback Comparison function
/// @param userData User data to pass to the callback
void __moore_scoreboard_set_compare_callback(MooreScoreboardHandle scoreboard,
                                              MooreScoreboardCompareCallback callback,
                                              void *userData);

/// Set the mismatch callback for a scoreboard.
/// Called when a mismatch is detected.
/// @param scoreboard Handle to the scoreboard
/// @param callback Mismatch notification function
/// @param userData User data to pass to the callback
void __moore_scoreboard_set_mismatch_callback(MooreScoreboardHandle scoreboard,
                                               MooreScoreboardMismatchCallback callback,
                                               void *userData);

/// Get the name of a scoreboard.
/// @param scoreboard Handle to the scoreboard
/// @return The scoreboard name as a MooreString
MooreString __moore_scoreboard_get_name(MooreScoreboardHandle scoreboard);

//===----------------------------------------------------------------------===//
// Scoreboard Transaction Operations
//===----------------------------------------------------------------------===//

/// Add an expected transaction to the scoreboard.
/// Expected transactions come from the reference model.
/// @param scoreboard Handle to the scoreboard
/// @param transaction Pointer to the transaction data
/// @param transactionSize Size of the transaction in bytes
void __moore_scoreboard_add_expected(MooreScoreboardHandle scoreboard,
                                      void *transaction,
                                      int64_t transactionSize);

/// Add an actual transaction to the scoreboard.
/// Actual transactions come from the DUT monitor.
/// @param scoreboard Handle to the scoreboard
/// @param transaction Pointer to the transaction data
/// @param transactionSize Size of the transaction in bytes
void __moore_scoreboard_add_actual(MooreScoreboardHandle scoreboard,
                                    void *transaction,
                                    int64_t transactionSize);

/// Compare the next expected vs actual transactions (blocking).
/// Waits for both expected and actual transactions to be available.
/// @param scoreboard Handle to the scoreboard
/// @return MOORE_SCOREBOARD_MATCH if match, MOORE_SCOREBOARD_MISMATCH otherwise
MooreScoreboardCompareResult __moore_scoreboard_compare(MooreScoreboardHandle scoreboard);

/// Try to compare transactions (non-blocking).
/// Returns immediately if either FIFO is empty.
/// @param scoreboard Handle to the scoreboard
/// @return Comparison result, or MOORE_SCOREBOARD_TIMEOUT if no transactions available
MooreScoreboardCompareResult __moore_scoreboard_try_compare(MooreScoreboardHandle scoreboard);

/// Compare all pending transactions (non-blocking).
/// Compares as many transactions as possible without blocking.
/// @param scoreboard Handle to the scoreboard
/// @return Number of comparisons performed
int64_t __moore_scoreboard_compare_all(MooreScoreboardHandle scoreboard);

//===----------------------------------------------------------------------===//
// Scoreboard TLM Integration
//===----------------------------------------------------------------------===//

/// Get the expected transaction FIFO's analysis export.
/// Connect this to the reference model's analysis port.
/// @param scoreboard Handle to the scoreboard
/// @return Handle to the analysis export, or MOORE_TLM_INVALID_HANDLE on error
MooreTlmPortHandle __moore_scoreboard_get_expected_export(MooreScoreboardHandle scoreboard);

/// Get the actual transaction FIFO's analysis export.
/// Connect this to the DUT monitor's analysis port.
/// @param scoreboard Handle to the scoreboard
/// @return Handle to the analysis export, or MOORE_TLM_INVALID_HANDLE on error
MooreTlmPortHandle __moore_scoreboard_get_actual_export(MooreScoreboardHandle scoreboard);

//===----------------------------------------------------------------------===//
// Scoreboard Statistics and Reporting
//===----------------------------------------------------------------------===//

/// Get the number of matching comparisons.
/// @param scoreboard Handle to the scoreboard
/// @return Number of transactions that matched
int64_t __moore_scoreboard_get_match_count(MooreScoreboardHandle scoreboard);

/// Get the number of mismatching comparisons.
/// @param scoreboard Handle to the scoreboard
/// @return Number of transactions that did not match
int64_t __moore_scoreboard_get_mismatch_count(MooreScoreboardHandle scoreboard);

/// Get the number of pending expected transactions.
/// @param scoreboard Handle to the scoreboard
/// @return Number of expected transactions not yet compared
int64_t __moore_scoreboard_get_pending_expected(MooreScoreboardHandle scoreboard);

/// Get the number of pending actual transactions.
/// @param scoreboard Handle to the scoreboard
/// @return Number of actual transactions not yet compared
int64_t __moore_scoreboard_get_pending_actual(MooreScoreboardHandle scoreboard);

/// Check if all transactions have been compared (no pending).
/// @param scoreboard Handle to the scoreboard
/// @return 1 if no pending transactions, 0 otherwise
int32_t __moore_scoreboard_is_empty(MooreScoreboardHandle scoreboard);

/// Print a verification report for the scoreboard.
/// Outputs match/mismatch counts and any pending transactions.
/// @param scoreboard Handle to the scoreboard
void __moore_scoreboard_report(MooreScoreboardHandle scoreboard);

/// Get the pass/fail status based on scoreboard results.
/// A scoreboard passes if there are no mismatches and no pending transactions.
/// @param scoreboard Handle to the scoreboard
/// @return 1 if passed (no errors), 0 if failed
int32_t __moore_scoreboard_passed(MooreScoreboardHandle scoreboard);

/// Reset the scoreboard statistics and clear all pending transactions.
/// @param scoreboard Handle to the scoreboard
void __moore_scoreboard_reset(MooreScoreboardHandle scoreboard);

//===----------------------------------------------------------------------===//
// Scoreboard Debugging/Tracing
//===----------------------------------------------------------------------===//

/// Enable or disable scoreboard tracing.
/// When enabled, all scoreboard operations are logged for debugging.
/// @param enable 1 to enable, 0 to disable
void __moore_scoreboard_set_trace_enabled(int32_t enable);

/// Check if scoreboard tracing is enabled.
/// @return 1 if enabled, 0 otherwise
int32_t __moore_scoreboard_is_trace_enabled(void);

/// Print a summary of all scoreboards and their current state.
void __moore_scoreboard_print_summary(void);

/// Get global statistics about scoreboard operations.
/// @param totalScoreboards Output: total number of scoreboards created
/// @param totalComparisons Output: total number of comparisons performed
/// @param totalMatches Output: total number of matches across all scoreboards
/// @param totalMismatches Output: total number of mismatches across all scoreboards
void __moore_scoreboard_get_statistics(int64_t *totalScoreboards,
                                        int64_t *totalComparisons,
                                        int64_t *totalMatches,
                                        int64_t *totalMismatches);

//===----------------------------------------------------------------------===//
// UVM Register Abstraction Layer (RAL) Infrastructure
//===----------------------------------------------------------------------===//
//
// These functions implement the UVM Register Abstraction Layer (RAL) runtime
// infrastructure for register-based verification. RAL provides a standardized
// way to model, access, and verify hardware registers and memories.
//
// UVM RAL Architecture:
// - uvm_reg_block: Container for registers, register files, and sub-blocks
// - uvm_reg: Individual register model with fields
// - uvm_reg_field: Bit fields within a register
// - uvm_reg_map: Address map for register access (supports multiple maps)
//
// Access Modes:
// - Frontdoor: Access through bus interface (uses sequences/drivers)
// - Backdoor: Direct access bypassing bus (faster, for debug)
//
// Key Features:
// - Mirror/desired value tracking for each register
// - Read/write operations with automatic prediction
// - Coverage collection for register access patterns
//
// Usage:
//   // Create register block and map
//   MooreRegBlockHandle blk = __moore_reg_block_create("my_block", 10);
//   MooreRegMapHandle map = __moore_reg_map_create(blk, "default_map", 14, 0,
//                                                   0x1000, 4, 0);
//
//   // Create and add registers
//   MooreRegHandle reg = __moore_reg_create("ctrl_reg", 32, 8);
//   __moore_reg_block_add_reg(blk, reg, 0x0000);
//
//   // Add register to map
//   __moore_reg_map_add_reg(map, reg, 0x0000, "RW");
//
//   // Access register
//   __moore_reg_write(reg, map, 0xDEADBEEF, UVM_FRONTDOOR, NULL, 0);
//   uint64_t value = __moore_reg_read(reg, map, UVM_FRONTDOOR, NULL, 0);
//

/// Handle type for UVM registers.
typedef int64_t MooreRegHandle;

/// Handle type for UVM register blocks.
typedef int64_t MooreRegBlockHandle;

/// Handle type for UVM register maps.
typedef int64_t MooreRegMapHandle;

/// Handle type for UVM register fields.
typedef int64_t MooreRegFieldHandle;

/// Invalid handle value for RAL components.
#define MOORE_REG_INVALID_HANDLE (-1)

/// UVM register access modes (from uvm_reg_model.svh).
typedef enum {
  UVM_FRONTDOOR = 0, ///< Access via bus interface (uses adapter/sequencer)
  UVM_BACKDOOR = 1,  ///< Direct access bypassing bus (hdl path)
  UVM_PREDICT = 2,   ///< Predict only, no actual access
  UVM_DEFAULT_PATH = 3 ///< Use default path set on register/block
} MooreRegPathKind;

/// UVM register field access policies.
typedef enum {
  UVM_REG_ACCESS_RO = 0,   ///< Read-only
  UVM_REG_ACCESS_RW = 1,   ///< Read-write
  UVM_REG_ACCESS_RC = 2,   ///< Read-clear (read returns value, clears to 0)
  UVM_REG_ACCESS_RS = 3,   ///< Read-set (read returns value, sets to all 1s)
  UVM_REG_ACCESS_WRC = 4,  ///< Write-1-to-clear, read returns value
  UVM_REG_ACCESS_WRS = 5,  ///< Write-1-to-set, read returns value
  UVM_REG_ACCESS_WC = 6,   ///< Write clears all bits
  UVM_REG_ACCESS_WS = 7,   ///< Write sets all bits
  UVM_REG_ACCESS_WSRC = 8, ///< Write sets all, read clears
  UVM_REG_ACCESS_WCRS = 9, ///< Write clears all, read sets
  UVM_REG_ACCESS_W1 = 10,  ///< Write once (first write only)
  UVM_REG_ACCESS_WO = 11,  ///< Write-only
  UVM_REG_ACCESS_WOC = 12, ///< Write-only clears
  UVM_REG_ACCESS_WOS = 13, ///< Write-only sets
  UVM_REG_ACCESS_W1C = 14, ///< Write-1-to-clear
  UVM_REG_ACCESS_W1S = 15, ///< Write-1-to-set
  UVM_REG_ACCESS_W1T = 16, ///< Write-1-to-toggle
  UVM_REG_ACCESS_W0C = 17, ///< Write-0-to-clear
  UVM_REG_ACCESS_W0S = 18, ///< Write-0-to-set
  UVM_REG_ACCESS_W0T = 19, ///< Write-0-to-toggle
  UVM_REG_ACCESS_W1SRC = 20, ///< Write-1-to-set, read clears
  UVM_REG_ACCESS_W1CRS = 21, ///< Write-1-to-clear, read sets
  UVM_REG_ACCESS_W0SRC = 22, ///< Write-0-to-set, read clears
  UVM_REG_ACCESS_W0CRS = 23, ///< Write-0-to-clear, read sets
  UVM_REG_ACCESS_WO1 = 24    ///< Write-once (any subsequent writes ignored)
} MooreRegAccessPolicy;

/// UVM register status from operations.
typedef enum {
  UVM_REG_STATUS_OK = 0,     ///< Operation completed successfully
  UVM_REG_STATUS_NOT_OK = 1, ///< Operation failed
  UVM_REG_STATUS_IS_BUSY = 2 ///< Register is busy (retry later)
} MooreRegStatus;

/// Register access callback function type.
/// Called before/after register read/write operations.
typedef void (*MooreRegAccessCallback)(MooreRegHandle reg, uint64_t value,
                                       int32_t isWrite, void *userData);

//===----------------------------------------------------------------------===//
// Register Operations
//===----------------------------------------------------------------------===//

/// Create a new UVM register model.
/// This creates an individual register with the specified name and size.
/// Fields can be added after creation using __moore_reg_add_field.
///
/// @param name Name of the register
/// @param nameLen Length of the name string
/// @param numBits Width of the register in bits (1-64)
/// @return Handle to the created register, or MOORE_REG_INVALID_HANDLE on failure
MooreRegHandle __moore_reg_create(const char *name, int64_t nameLen,
                                  int32_t numBits);

/// Destroy a register and free its resources.
/// @param reg Handle to the register to destroy
void __moore_reg_destroy(MooreRegHandle reg);

/// Get the name of a register.
/// @param reg Handle to the register
/// @return The register name as a MooreString (caller must free)
MooreString __moore_reg_get_name(MooreRegHandle reg);

/// Get the bit width of a register.
/// @param reg Handle to the register
/// @return Width in bits (1-64)
int32_t __moore_reg_get_n_bits(MooreRegHandle reg);

/// Get the address of a register within a specific map.
/// @param reg Handle to the register
/// @param map Handle to the address map (or MOORE_REG_INVALID_HANDLE for default)
/// @return The register's base address in the specified map
uint64_t __moore_reg_get_address(MooreRegHandle reg, MooreRegMapHandle map);

/// Read the current value from a register.
/// This performs a read operation using the specified access path.
/// For frontdoor access, this would use the bus adapter/sequencer.
/// For backdoor access, this reads the HDL signal directly.
///
/// @param reg Handle to the register
/// @param map Handle to the address map (NULL for default)
/// @param path Access path (UVM_FRONTDOOR, UVM_BACKDOOR, etc.)
/// @param status Output: status of the operation (can be NULL)
/// @param parent Sequencer parent for frontdoor access (can be 0)
/// @return The value read from the register
uint64_t __moore_reg_read(MooreRegHandle reg, MooreRegMapHandle map,
                          MooreRegPathKind path, MooreRegStatus *status,
                          int64_t parent);

/// Write a value to a register.
/// This performs a write operation using the specified access path.
/// For frontdoor access, this would use the bus adapter/sequencer.
/// For backdoor access, this writes the HDL signal directly.
///
/// @param reg Handle to the register
/// @param map Handle to the address map (NULL for default)
/// @param value Value to write
/// @param path Access path (UVM_FRONTDOOR, UVM_BACKDOOR, etc.)
/// @param status Output: status of the operation (can be NULL)
/// @param parent Sequencer parent for frontdoor access (can be 0)
void __moore_reg_write(MooreRegHandle reg, MooreRegMapHandle map,
                       uint64_t value, MooreRegPathKind path,
                       MooreRegStatus *status, int64_t parent);

/// Get the mirrored (predicted) value of a register.
/// The mirror value tracks what the register value should be based on
/// read/write operations and reset.
///
/// @param reg Handle to the register
/// @return The current mirror value
uint64_t __moore_reg_get_value(MooreRegHandle reg);

/// Set the mirror value of a register directly.
/// This updates the predicted value without performing an actual access.
/// Useful for initializing register models or correcting predictions.
///
/// @param reg Handle to the register
/// @param value New mirror value
void __moore_reg_set_value(MooreRegHandle reg, uint64_t value);

/// Get the desired value of a register.
/// The desired value is the value intended to be written on next update.
///
/// @param reg Handle to the register
/// @return The current desired value
uint64_t __moore_reg_get_desired(MooreRegHandle reg);

/// Set the desired value of a register.
/// This sets the value that will be written on the next update operation.
///
/// @param reg Handle to the register
/// @param value New desired value
void __moore_reg_set_desired(MooreRegHandle reg, uint64_t value);

/// Update a register (write desired value to actual).
/// Writes the current desired value to the register using the specified path.
///
/// @param reg Handle to the register
/// @param map Handle to the address map
/// @param path Access path (UVM_FRONTDOOR, UVM_BACKDOOR)
/// @param status Output: status of the operation (can be NULL)
void __moore_reg_update(MooreRegHandle reg, MooreRegMapHandle map,
                        MooreRegPathKind path, MooreRegStatus *status);

/// Mirror a register (read actual value into mirror).
/// Reads the register and updates the mirror to match the actual value.
///
/// @param reg Handle to the register
/// @param map Handle to the address map
/// @param path Access path (UVM_FRONTDOOR, UVM_BACKDOOR)
/// @param status Output: status of the operation (can be NULL)
void __moore_reg_mirror(MooreRegHandle reg, MooreRegMapHandle map,
                        MooreRegPathKind path, MooreRegStatus *status);

/// Predict the effect of a read or write on the register mirror.
/// Updates the mirror value based on the access policy without actual access.
///
/// @param reg Handle to the register
/// @param value Value being read/written
/// @param isWrite true for write prediction, false for read prediction
/// @return true if prediction succeeded
bool __moore_reg_predict(MooreRegHandle reg, uint64_t value, bool isWrite);

/// Reset the register to its reset value.
/// @param reg Handle to the register
/// @param kind Reset kind: "HARD" for power-on, "SOFT" for soft reset
void __moore_reg_reset(MooreRegHandle reg, const char *kind);

/// Set the reset value for a register.
/// @param reg Handle to the register
/// @param value Reset value
/// @param kind Reset kind ("HARD" or "SOFT")
void __moore_reg_set_reset(MooreRegHandle reg, uint64_t value, const char *kind);

/// Get the reset value for a register.
/// @param reg Handle to the register
/// @param kind Reset kind ("HARD" or "SOFT")
/// @return The reset value
uint64_t __moore_reg_get_reset(MooreRegHandle reg, const char *kind);

/// Check if mirror needs to be updated (value changed).
/// @param reg Handle to the register
/// @return true if mirror differs from desired
bool __moore_reg_needs_update(MooreRegHandle reg);

/// Set a callback for register access events.
/// @param reg Handle to the register
/// @param callback Function to call on access (NULL to disable)
/// @param userData User data passed to callback
void __moore_reg_set_access_callback(MooreRegHandle reg,
                                     MooreRegAccessCallback callback,
                                     void *userData);

//===----------------------------------------------------------------------===//
// Register Field Operations
//===----------------------------------------------------------------------===//

/// Add a field to a register.
/// Fields represent named bit ranges within a register with specific access
/// policies.
///
/// @param reg Handle to the register
/// @param name Field name
/// @param nameLen Length of the name string
/// @param numBits Width of the field in bits
/// @param lsbPos LSB position of the field within the register
/// @param access Access policy for the field
/// @param reset Reset value for the field
/// @return Handle to the created field, or MOORE_REG_INVALID_HANDLE on failure
MooreRegFieldHandle __moore_reg_add_field(MooreRegHandle reg, const char *name,
                                          int64_t nameLen, int32_t numBits,
                                          int32_t lsbPos,
                                          MooreRegAccessPolicy access,
                                          uint64_t reset);

/// Get the value of a specific field within a register.
/// @param reg Handle to the register
/// @param field Handle to the field
/// @return The field value (extracted from mirror)
uint64_t __moore_reg_field_get_value(MooreRegHandle reg,
                                     MooreRegFieldHandle field);

/// Set the value of a specific field within a register.
/// Updates the mirror value for just this field.
/// @param reg Handle to the register
/// @param field Handle to the field
/// @param value New field value
void __moore_reg_field_set_value(MooreRegHandle reg, MooreRegFieldHandle field,
                                 uint64_t value);

/// Get a field handle by name.
/// @param reg Handle to the register
/// @param name Field name to search for
/// @param nameLen Length of the name string
/// @return Handle to the field, or MOORE_REG_INVALID_HANDLE if not found
MooreRegFieldHandle __moore_reg_get_field_by_name(MooreRegHandle reg,
                                                  const char *name,
                                                  int64_t nameLen);

/// Get the number of fields in a register.
/// @param reg Handle to the register
/// @return Number of fields
int32_t __moore_reg_get_n_fields(MooreRegHandle reg);

//===----------------------------------------------------------------------===//
// Register Block Operations
//===----------------------------------------------------------------------===//

/// Create a new register block.
/// A register block is a container that groups registers, register files,
/// and sub-blocks together with a common address map.
///
/// @param name Name of the register block
/// @param nameLen Length of the name string
/// @return Handle to the created block, or MOORE_REG_INVALID_HANDLE on failure
MooreRegBlockHandle __moore_reg_block_create(const char *name, int64_t nameLen);

/// Destroy a register block and all its contents.
/// This recursively destroys all registers, fields, maps, and sub-blocks.
/// @param block Handle to the block to destroy
void __moore_reg_block_destroy(MooreRegBlockHandle block);

/// Get the name of a register block.
/// @param block Handle to the register block
/// @return The block name as a MooreString (caller must free)
MooreString __moore_reg_block_get_name(MooreRegBlockHandle block);

/// Add a register to a block.
/// @param block Handle to the register block
/// @param reg Handle to the register to add
/// @param offset Address offset of the register within the block
void __moore_reg_block_add_reg(MooreRegBlockHandle block, MooreRegHandle reg,
                               uint64_t offset);

/// Add a sub-block to a block.
/// Creates a hierarchical register model.
/// @param parent Handle to the parent block
/// @param child Handle to the child block to add
/// @param offset Base address offset for the sub-block
void __moore_reg_block_add_block(MooreRegBlockHandle parent,
                                 MooreRegBlockHandle child, uint64_t offset);

/// Get the default map for a register block.
/// @param block Handle to the register block
/// @return Handle to the default map, or MOORE_REG_INVALID_HANDLE if none
MooreRegMapHandle __moore_reg_block_get_default_map(MooreRegBlockHandle block);

/// Set the default map for a register block.
/// @param block Handle to the register block
/// @param map Handle to the map to set as default
void __moore_reg_block_set_default_map(MooreRegBlockHandle block,
                                       MooreRegMapHandle map);

/// Get a register by name from a block.
/// Searches the block hierarchy for a register with the given name.
/// @param block Handle to the register block
/// @param name Register name (may include hierarchy: "subblock.reg")
/// @param nameLen Length of the name string
/// @return Handle to the register, or MOORE_REG_INVALID_HANDLE if not found
MooreRegHandle __moore_reg_block_get_reg_by_name(MooreRegBlockHandle block,
                                                 const char *name,
                                                 int64_t nameLen);

/// Get the number of registers in a block (non-recursive).
/// @param block Handle to the register block
/// @return Number of registers directly in this block
int32_t __moore_reg_block_get_n_regs(MooreRegBlockHandle block);

/// Lock the register block model.
/// After locking, no structural changes (adding regs/fields/maps) are allowed.
/// This is typically called after building the complete register model.
/// @param block Handle to the register block
void __moore_reg_block_lock(MooreRegBlockHandle block);

/// Check if a register block is locked.
/// @param block Handle to the register block
/// @return true if locked
bool __moore_reg_block_is_locked(MooreRegBlockHandle block);

/// Reset all registers in a block to their reset values.
/// @param block Handle to the register block
/// @param kind Reset kind: "HARD" for power-on, "SOFT" for soft reset
void __moore_reg_block_reset(MooreRegBlockHandle block, const char *kind);

//===----------------------------------------------------------------------===//
// Register Map Operations
//===----------------------------------------------------------------------===//

/// Create a new register map.
/// A register map defines address decoding for a set of registers and provides
/// a transaction interface (sequencer/adapter) for frontdoor access.
///
/// @param block Handle to the parent register block
/// @param name Name of the map
/// @param nameLen Length of the name string
/// @param baseAddr Base address for the map
/// @param nBytes Bus width in bytes (e.g., 4 for 32-bit bus)
/// @param endian Endianness (0=little, 1=big)
/// @return Handle to the created map, or MOORE_REG_INVALID_HANDLE on failure
MooreRegMapHandle __moore_reg_map_create(MooreRegBlockHandle block,
                                         const char *name, int64_t nameLen,
                                         uint64_t baseAddr, int32_t nBytes,
                                         int32_t endian);

/// Destroy a register map.
/// @param map Handle to the map to destroy
void __moore_reg_map_destroy(MooreRegMapHandle map);

/// Get the name of a register map.
/// @param map Handle to the register map
/// @return The map name as a MooreString (caller must free)
MooreString __moore_reg_map_get_name(MooreRegMapHandle map);

/// Get the base address of a register map.
/// @param map Handle to the register map
/// @return The base address
uint64_t __moore_reg_map_get_base_addr(MooreRegMapHandle map);

/// Add a register to a map with specific access rights.
/// This maps a register to an address within the map's address space.
///
/// @param map Handle to the register map
/// @param reg Handle to the register to add
/// @param offset Address offset from map's base address
/// @param rights Access rights string ("RW", "RO", "WO", etc.)
void __moore_reg_map_add_reg(MooreRegMapHandle map, MooreRegHandle reg,
                             uint64_t offset, const char *rights);

/// Add a sub-map to a map.
/// Creates a hierarchical address space.
/// @param parent Handle to the parent map
/// @param child Handle to the child map to add
/// @param offset Address offset for the sub-map
void __moore_reg_map_add_submap(MooreRegMapHandle parent,
                                MooreRegMapHandle child, uint64_t offset);

/// Get a register by address from a map.
/// @param map Handle to the register map
/// @param addr Address to look up
/// @return Handle to the register at that address, or MOORE_REG_INVALID_HANDLE
MooreRegHandle __moore_reg_map_get_reg_by_addr(MooreRegMapHandle map,
                                               uint64_t addr);

/// Get the offset of a register within a map.
/// @param map Handle to the register map
/// @param reg Handle to the register
/// @return The register's offset within the map
uint64_t __moore_reg_map_get_reg_offset(MooreRegMapHandle map,
                                        MooreRegHandle reg);

/// Set the sequencer for frontdoor access.
/// The sequencer is used for generating bus transactions.
/// @param map Handle to the register map
/// @param sequencer Handle to the sequencer (from sequence infrastructure)
void __moore_reg_map_set_sequencer(MooreRegMapHandle map, int64_t sequencer);

/// Set the bus adapter for translating register operations to bus transactions.
/// @param map Handle to the register map
/// @param adapter Adapter handle (implementation-specific)
void __moore_reg_map_set_adapter(MooreRegMapHandle map, int64_t adapter);

//===----------------------------------------------------------------------===//
// RAL Debugging and Tracing
//===----------------------------------------------------------------------===//

/// Enable or disable RAL tracing.
/// When enabled, all register operations are logged for debugging.
/// @param enable 1 to enable, 0 to disable
void __moore_reg_set_trace_enabled(int32_t enable);

/// Check if RAL tracing is enabled.
/// @return 1 if enabled, 0 otherwise
int32_t __moore_reg_is_trace_enabled(void);

/// Print a summary of the register model hierarchy.
/// Outputs block/register/field structure to stdout.
/// @param block Handle to the register block (root)
void __moore_reg_block_print(MooreRegBlockHandle block);

/// Get global RAL statistics.
/// @param totalRegs Output: total number of registers created
/// @param totalReads Output: total read operations performed
/// @param totalWrites Output: total write operations performed
void __moore_reg_get_statistics(int64_t *totalRegs, int64_t *totalReads,
                                int64_t *totalWrites);

/// Clear all RAL components and reset statistics.
void __moore_reg_clear_all(void);

//===----------------------------------------------------------------------===//
// UVM Message Reporting Infrastructure
//===----------------------------------------------------------------------===//
//
// These functions implement UVM-compatible message reporting with verbosity
// filtering, severity tracking, and formatted output. They provide the runtime
// support for UVM_INFO, UVM_WARNING, UVM_ERROR, and UVM_FATAL macros.
//
// UVM Severity Levels:
// - UVM_INFO: Informational message (filtered by verbosity)
// - UVM_WARNING: Warning condition (always displayed, tracked)
// - UVM_ERROR: Error condition (always displayed, tracked, may terminate)
// - UVM_FATAL: Fatal error (always displayed, terminates simulation)
//
// UVM Verbosity Levels (from IEEE 1800.2):
// - UVM_NONE (0): Always display
// - UVM_LOW (100): Low verbosity
// - UVM_MEDIUM (200): Medium verbosity (default)
// - UVM_HIGH (300): High verbosity
// - UVM_FULL (400): Full verbosity
// - UVM_DEBUG (500): Debug verbosity
//
// Messages are displayed if: message_verbosity <= report_verbosity_threshold
//

/// UVM severity levels (matches uvm_severity enum).
typedef enum {
  MOORE_UVM_INFO = 0,
  MOORE_UVM_WARNING = 1,
  MOORE_UVM_ERROR = 2,
  MOORE_UVM_FATAL = 3
} MooreUvmSeverity;

/// UVM verbosity levels (matches uvm_verbosity enum values).
typedef enum {
  MOORE_UVM_NONE = 0,
  MOORE_UVM_LOW = 100,
  MOORE_UVM_MEDIUM = 200,
  MOORE_UVM_HIGH = 300,
  MOORE_UVM_FULL = 400,
  MOORE_UVM_DEBUG = 500
} MooreUvmVerbosity;

/// UVM action types (matches uvm_action enum).
/// Actions can be combined using bitwise OR.
typedef enum {
  MOORE_UVM_NO_ACTION = 0,
  MOORE_UVM_DISPLAY = (1 << 0),
  MOORE_UVM_LOG = (1 << 1),
  MOORE_UVM_COUNT = (1 << 2),
  MOORE_UVM_EXIT = (1 << 3),
  MOORE_UVM_CALL_HOOK = (1 << 4),
  MOORE_UVM_STOP = (1 << 5),
  MOORE_UVM_RM_RECORD = (1 << 6)
} MooreUvmAction;

/// Set the global verbosity threshold for UVM reporting.
/// Messages with verbosity level above this threshold are suppressed.
/// Default is MOORE_UVM_MEDIUM (200).
///
/// @param verbosity The verbosity threshold to set
void __moore_uvm_set_report_verbosity(int32_t verbosity);

/// Get the current global verbosity threshold.
///
/// @return The current verbosity threshold
int32_t __moore_uvm_get_report_verbosity(void);

/// Report a UVM informational message.
/// The message is displayed if its verbosity level is <= the current threshold.
/// UVM_INFO messages are printed to stdout with format:
///   UVM_INFO <filename>(<line>) @ <time>: <id> [<context>] <message>
///
/// @param id The message ID (e.g., "MYTEST", "DRIVER")
/// @param idLen Length of the id string
/// @param message The message text
/// @param messageLen Length of the message string
/// @param verbosity The verbosity level of this message
/// @param filename Source filename (may be NULL)
/// @param filenameLen Length of filename string
/// @param line Source line number
/// @param context Hierarchical context (may be NULL)
/// @param contextLen Length of context string
void __moore_uvm_report_info(const char *id, int64_t idLen, const char *message,
                             int64_t messageLen, int32_t verbosity,
                             const char *filename, int64_t filenameLen,
                             int32_t line, const char *context,
                             int64_t contextLen);

/// Report a UVM warning message.
/// Warnings are always displayed (not filtered by verbosity).
/// UVM_WARNING messages are printed to stderr with format:
///   UVM_WARNING <filename>(<line>) @ <time>: <id> [<context>] <message>
/// Increments the warning count.
///
/// @param id The message ID
/// @param idLen Length of the id string
/// @param message The message text
/// @param messageLen Length of the message string
/// @param verbosity The verbosity level (usually UVM_NONE)
/// @param filename Source filename (may be NULL)
/// @param filenameLen Length of filename string
/// @param line Source line number
/// @param context Hierarchical context (may be NULL)
/// @param contextLen Length of context string
void __moore_uvm_report_warning(const char *id, int64_t idLen,
                                const char *message, int64_t messageLen,
                                int32_t verbosity, const char *filename,
                                int64_t filenameLen, int32_t line,
                                const char *context, int64_t contextLen);

/// Report a UVM error message.
/// Errors are always displayed (not filtered by verbosity).
/// UVM_ERROR messages are printed to stderr with format:
///   UVM_ERROR <filename>(<line>) @ <time>: <id> [<context>] <message>
/// Increments the error count. May terminate simulation if max_quit_count
/// is exceeded.
///
/// @param id The message ID
/// @param idLen Length of the id string
/// @param message The message text
/// @param messageLen Length of the message string
/// @param verbosity The verbosity level (usually UVM_NONE)
/// @param filename Source filename (may be NULL)
/// @param filenameLen Length of filename string
/// @param line Source line number
/// @param context Hierarchical context (may be NULL)
/// @param contextLen Length of context string
void __moore_uvm_report_error(const char *id, int64_t idLen,
                              const char *message, int64_t messageLen,
                              int32_t verbosity, const char *filename,
                              int64_t filenameLen, int32_t line,
                              const char *context, int64_t contextLen);

/// Report a UVM fatal error message and terminate simulation.
/// UVM_FATAL messages are printed to stderr with format:
///   UVM_FATAL <filename>(<line>) @ <time>: <id> [<context>] <message>
/// Always terminates simulation with exit code 1.
///
/// @param id The message ID
/// @param idLen Length of the id string
/// @param message The message text
/// @param messageLen Length of the message string
/// @param verbosity The verbosity level (usually UVM_NONE)
/// @param filename Source filename (may be NULL)
/// @param filenameLen Length of filename string
/// @param line Source line number
/// @param context Hierarchical context (may be NULL)
/// @param contextLen Length of context string
void __moore_uvm_report_fatal(const char *id, int64_t idLen,
                              const char *message, int64_t messageLen,
                              int32_t verbosity, const char *filename,
                              int64_t filenameLen, int32_t line,
                              const char *context, int64_t contextLen);

/// Check if a message with given severity, verbosity, and ID should be reported.
/// This implements UVM's uvm_report_enabled() functionality.
///
/// @param verbosity The verbosity level of the message
/// @param severity The severity level of the message
/// @param id The message ID (for ID-specific verbosity settings)
/// @param idLen Length of the id string
/// @return 1 if the message should be reported, 0 otherwise
int32_t __moore_uvm_report_enabled(int32_t verbosity, int32_t severity,
                                   const char *id, int64_t idLen);

/// Set the verbosity threshold for a specific message ID.
/// This allows fine-grained control over which messages are displayed.
///
/// @param id The message ID to configure
/// @param idLen Length of the id string
/// @param verbosity The verbosity threshold for this ID
void __moore_uvm_set_report_id_verbosity(const char *id, int64_t idLen,
                                         int32_t verbosity);

/// Get the UVM message count for a specific severity.
///
/// @param severity The severity level to query
/// @return Number of messages reported at that severity
int32_t __moore_uvm_get_report_count(int32_t severity);

/// Reset all UVM message counts to zero.
void __moore_uvm_reset_report_counts(void);

/// Set the maximum number of errors before simulation terminates.
/// When the error count reaches this value, simulation exits.
/// Default is 0 (unlimited errors).
///
/// @param count Maximum error count (0 = unlimited)
void __moore_uvm_set_max_quit_count(int32_t count);

/// Get the maximum quit count setting.
///
/// @return Current maximum quit count (0 = unlimited)
int32_t __moore_uvm_get_max_quit_count(void);

/// Set the default action for a severity level.
/// The default actions are:
/// - UVM_INFO: DISPLAY
/// - UVM_WARNING: DISPLAY | COUNT
/// - UVM_ERROR: DISPLAY | COUNT
/// - UVM_FATAL: DISPLAY | EXIT
///
/// @param severity The severity level to configure
/// @param action Bitmask of MooreUvmAction values
void __moore_uvm_set_report_severity_action(int32_t severity, int32_t action);

/// Get the current action for a severity level.
///
/// @param severity The severity level to query
/// @return Bitmask of current actions
int32_t __moore_uvm_get_report_severity_action(int32_t severity);

/// Print a summary of UVM messages reported during simulation.
/// Output includes counts for each severity level and overall status.
void __moore_uvm_report_summarize(void);

/// Set whether UVM_FATAL should actually exit or just set a flag.
/// This is useful for testing the runtime functions.
///
/// @param should_exit true to exit on fatal, false to just set flag
void __moore_uvm_set_fatal_exits(bool should_exit);

/// Get the current simulation time for message timestamps.
/// This interfaces with the simulation time tracking infrastructure.
///
/// @return Current simulation time in simulation units
uint64_t __moore_uvm_get_time(void);

/// Set the simulation time (for use by the simulation infrastructure).
///
/// @param time Current simulation time
void __moore_uvm_set_time(uint64_t time);

//===----------------------------------------------------------------------===//
// UVM Root Re-entrancy Support
//===----------------------------------------------------------------------===//
//
// These functions help handle the re-entrancy issue in UVM where
// uvm_component::new() calls get_root(), but during uvm_root::new(),
// m_inst is set before uvm_top. This causes a false mismatch warning.
//
// The fix works by:
// 1. Setting a flag when m_uvm_get_root() starts constructing uvm_root
// 2. Re-entrant calls to get_root() during construction return m_inst directly
// 3. The flag is cleared after construction completes
//

/// Mark that uvm_root construction has started.
/// This should be called at the beginning of m_uvm_get_root() before
/// calling uvm_root::new(). It sets a flag to indicate that re-entrant
/// calls to get_root() should return m_inst directly without comparing
/// to uvm_top (which isn't set yet).
void __moore_uvm_root_constructing_start(void);

/// Mark that uvm_root construction has completed.
/// This should be called after m_uvm_get_root() finishes setting uvm_top.
/// It clears the construction-in-progress flag.
void __moore_uvm_root_constructing_end(void);

/// Check if uvm_root is currently being constructed.
/// Used by re-entrant code paths to skip the m_inst != uvm_top check.
///
/// @return true if root construction is in progress, false otherwise
bool __moore_uvm_is_root_constructing(void);

/// Set the uvm_root instance pointer during construction.
/// This is called to set m_inst atomically with the constructing flag.
///
/// @param inst Pointer to the uvm_root instance being constructed
void __moore_uvm_set_root_inst(void *inst);

/// Get the uvm_root instance pointer.
/// During construction, returns the partially constructed instance.
///
/// @return Pointer to the uvm_root instance (may be partially constructed)
void *__moore_uvm_get_root_inst(void);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_RUNTIME_MOORERUNTIME_H
