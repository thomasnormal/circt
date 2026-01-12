// RUN: circt-opt --verify-diagnostics --verify-roundtrip %s | FileCheck %s

//===----------------------------------------------------------------------===//
// String Concatenation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @StringConcat
// CHECK-SAME: ([[S1:%.+]]: !moore.string, [[S2:%.+]]: !moore.string, [[S3:%.+]]: !moore.string)
func.func @StringConcat(%s1: !moore.string, %s2: !moore.string, %s3: !moore.string) {
  // CHECK: moore.string_concat () : string
  %empty = moore.string_concat () : !moore.string
  // CHECK: moore.string_concat ([[S1]]) : string
  %single = moore.string_concat (%s1) : !moore.string
  // CHECK: moore.string_concat ([[S1]], [[S2]]) : string
  %two = moore.string_concat (%s1, %s2) : !moore.string
  // CHECK: moore.string_concat ([[S1]], [[S2]], [[S3]]) : string
  %three = moore.string_concat (%s1, %s2, %s3) : !moore.string
  return
}

//===----------------------------------------------------------------------===//
// Queue Push/Pop Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @QueuePushPop
// CHECK-SAME: ([[QUEUE:%.+]]: !moore.ref<!moore.queue<i32, 0>>, [[ELEM:%.+]]: !moore.i32)
func.func @QueuePushPop(%queue: !moore.ref<!moore.queue<i32, 0>>, %elem: !moore.i32) {
  // CHECK: moore.queue.push_back [[QUEUE]], [[ELEM]] : <!moore.queue<i32, 0>>, i32
  moore.queue.push_back %queue, %elem : <!moore.queue<i32, 0>>, i32
  // CHECK: moore.queue.push_front [[QUEUE]], [[ELEM]] : <!moore.queue<i32, 0>>, i32
  moore.queue.push_front %queue, %elem : <!moore.queue<i32, 0>>, i32
  // CHECK: moore.queue.pop_back [[QUEUE]] : <!moore.queue<i32, 0>> -> i32
  %back = moore.queue.pop_back %queue : <!moore.queue<i32, 0>> -> i32
  // CHECK: moore.queue.pop_front [[QUEUE]] : <!moore.queue<i32, 0>> -> i32
  %front = moore.queue.pop_front %queue : <!moore.queue<i32, 0>> -> i32
  return
}

//===----------------------------------------------------------------------===//
// Queue Sort and Delete Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @QueueSortDelete
// CHECK-SAME: ([[QUEUE:%.+]]: !moore.ref<!moore.queue<i32, 0>>)
func.func @QueueSortDelete(%queue: !moore.ref<!moore.queue<i32, 0>>) {
  // CHECK: moore.queue.sort [[QUEUE]] : <!moore.queue<i32, 0>>
  moore.queue.sort %queue : <!moore.queue<i32, 0>>
  // CHECK: moore.queue.delete [[QUEUE]] : <!moore.queue<i32, 0>>
  moore.queue.delete %queue : <!moore.queue<i32, 0>>
  return
}

// Test with unbounded queue (bound = 0 indicates no size limit)
// CHECK-LABEL: func.func @QueueUnbounded
// CHECK-SAME: ([[QUEUE:%.+]]: !moore.ref<!moore.queue<string, 0>>, [[ELEM:%.+]]: !moore.string)
func.func @QueueUnbounded(%queue: !moore.ref<!moore.queue<string, 0>>, %elem: !moore.string) {
  // CHECK: moore.queue.push_back [[QUEUE]], [[ELEM]] : <!moore.queue<string, 0>>, string
  moore.queue.push_back %queue, %elem : <!moore.queue<string, 0>>, string
  // CHECK: moore.queue.pop_front [[QUEUE]] : <!moore.queue<string, 0>> -> string
  %front = moore.queue.pop_front %queue : <!moore.queue<string, 0>> -> string
  // CHECK: moore.queue.sort [[QUEUE]] : <!moore.queue<string, 0>>
  moore.queue.sort %queue : <!moore.queue<string, 0>>
  // CHECK: moore.queue.delete [[QUEUE]] : <!moore.queue<string, 0>>
  moore.queue.delete %queue : <!moore.queue<string, 0>>
  return
}

//===----------------------------------------------------------------------===//
// Associative Array Iteration Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @AssocArrayIteration
// CHECK-SAME: ([[ARRAY:%.+]]: !moore.ref<!moore.assoc_array<i32, string>>, [[KEY:%.+]]: !moore.ref<!moore.string>)
func.func @AssocArrayIteration(%array: !moore.ref<!moore.assoc_array<i32, string>>, %key: !moore.ref<!moore.string>) {
  // CHECK: moore.assoc.first [[ARRAY]], [[KEY]] : <!moore.assoc_array<i32, string>>, <!moore.string>
  %found_first = moore.assoc.first %array, %key : <!moore.assoc_array<i32, string>>, <!moore.string>
  // CHECK: moore.assoc.next [[ARRAY]], [[KEY]] : <!moore.assoc_array<i32, string>>, <!moore.string>
  %found_next = moore.assoc.next %array, %key : <!moore.assoc_array<i32, string>>, <!moore.string>
  return
}

// Test associative array with different key/value types
// CHECK-LABEL: func.func @AssocArrayWithIntKey
// CHECK-SAME: ([[ARRAY:%.+]]: !moore.ref<!moore.assoc_array<string, i64>>, [[KEY:%.+]]: !moore.ref<!moore.i64>)
func.func @AssocArrayWithIntKey(%array: !moore.ref<!moore.assoc_array<string, i64>>, %key: !moore.ref<!moore.i64>) {
  // CHECK: moore.assoc.first [[ARRAY]], [[KEY]] : <!moore.assoc_array<string, i64>>, <!moore.i64>
  %found_first = moore.assoc.first %array, %key : <!moore.assoc_array<string, i64>>, <!moore.i64>
  // CHECK: moore.assoc.next [[ARRAY]], [[KEY]] : <!moore.assoc_array<string, i64>>, <!moore.i64>
  %found_next = moore.assoc.next %array, %key : <!moore.assoc_array<string, i64>>, <!moore.i64>
  return
}
