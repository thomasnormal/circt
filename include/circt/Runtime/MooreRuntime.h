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

/// Slice a queue with an inclusive range [start, end].
/// @param queue Pointer to the queue structure
/// @param start Start index (inclusive)
/// @param end End index (inclusive)
/// @param element_size Size of each element in bytes
/// @return A new queue containing the selected slice
MooreQueue __moore_queue_slice(MooreQueue *queue, int64_t start, int64_t end,
                               int64_t element_size);

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

/// Create a new empty associative array.
/// @param key_size Size of keys in bytes (0 for string keys)
/// @param value_size Size of values in bytes
/// @return Pointer to the new associative array (opaque handle)
void *__moore_assoc_create(int32_t key_size, int32_t value_size);

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
///
/// @param cg Pointer to the covergroup
/// @return Coverage percentage (0.0 to 100.0)
double __moore_covergroup_get_coverage(void *cg);

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
  MOORE_BIN_WILDCARD = 2,  ///< Wildcard bin (future): bins x = {4'b1???};
  MOORE_BIN_TRANSITION = 3 ///< Transition bin (future): bins x = (1 => 2);
};

/// Coverage bin definition structure.
/// @member name Name of the bin
/// @member type Type of bin (value, range, wildcard, transition)
/// @member low Lower bound of range (or single value for value bins)
/// @member high Upper bound of range (same as low for value bins)
/// @member hit_count Number of times this bin was hit
typedef struct {
  const char *name;
  int32_t type;
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
// Memory Management
//===----------------------------------------------------------------------===//

/// Free memory allocated by the Moore runtime.
/// Should be called to release strings and arrays when no longer needed.
/// @param ptr Pointer to the memory to free
void __moore_free(void *ptr);

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
MooreString uvm_dpi_get_next_arg_c(int32_t *idx);

/// Get the name of the simulation tool.
/// @return String containing the tool name (stub: returns "CIRCT")
MooreString uvm_dpi_get_tool_name_c(void);

/// Get the version of the simulation tool.
/// @return String containing the tool version (stub: returns "1.0")
MooreString uvm_dpi_get_tool_version_c(void);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_RUNTIME_MOORERUNTIME_H
