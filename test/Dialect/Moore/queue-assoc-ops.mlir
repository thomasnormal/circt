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
// CHECK-SAME: ([[QUEUE:%.+]]: !moore.ref<queue<i32, 0>>, [[ELEM:%.+]]: !moore.i32)
func.func @QueuePushPop(%queue: !moore.ref<queue<i32, 0>>, %elem: !moore.i32) {
  // CHECK: moore.queue.push_back [[QUEUE]], [[ELEM]] : <queue<i32, 0>>, i32
  moore.queue.push_back %queue, %elem : <queue<i32, 0>>, i32
  // CHECK: moore.queue.push_front [[QUEUE]], [[ELEM]] : <queue<i32, 0>>, i32
  moore.queue.push_front %queue, %elem : <queue<i32, 0>>, i32
  // CHECK: moore.queue.pop_back [[QUEUE]] : <queue<i32, 0>> -> i32
  %back = moore.queue.pop_back %queue : <queue<i32, 0>> -> !moore.i32
  // CHECK: moore.queue.pop_front [[QUEUE]] : <queue<i32, 0>> -> i32
  %front = moore.queue.pop_front %queue : <queue<i32, 0>> -> !moore.i32
  return
}

// CHECK-LABEL: func.func @QueueInsert
// CHECK-SAME: ([[QUEUE:%.+]]: !moore.ref<queue<i32, 0>>, [[INDEX:%.+]]: !moore.i32, [[ELEM:%.+]]: !moore.i32)
func.func @QueueInsert(%queue: !moore.ref<queue<i32, 0>>, %index: !moore.i32, %elem: !moore.i32) {
  // CHECK: moore.queue.insert [[QUEUE]], [[INDEX]], [[ELEM]] : <queue<i32, 0>>, i32, i32
  moore.queue.insert %queue, %index, %elem : <queue<i32, 0>>, i32, i32
  return
}

//===----------------------------------------------------------------------===//
// Queue Sort and Delete Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @QueueSortDelete
// CHECK-SAME: ([[QUEUE:%.+]]: !moore.ref<queue<i32, 0>>)
func.func @QueueSortDelete(%queue: !moore.ref<queue<i32, 0>>) {
  // CHECK: moore.queue.sort [[QUEUE]] : <queue<i32, 0>>
  moore.queue.sort %queue : <queue<i32, 0>>
  // CHECK: moore.queue.delete [[QUEUE]] : <queue<i32, 0>>
  moore.queue.delete %queue : <queue<i32, 0>>
  return
}

// Test with unbounded queue (bound = 0 indicates no size limit)
// CHECK-LABEL: func.func @QueueUnbounded
// CHECK-SAME: ([[QUEUE:%.+]]: !moore.ref<queue<string, 0>>, [[ELEM:%.+]]: !moore.string)
func.func @QueueUnbounded(%queue: !moore.ref<queue<string, 0>>, %elem: !moore.string) {
  // CHECK: moore.queue.push_back [[QUEUE]], [[ELEM]] : <queue<string, 0>>, string
  moore.queue.push_back %queue, %elem : <queue<string, 0>>, string
  // CHECK: moore.queue.pop_front [[QUEUE]] : <queue<string, 0>> -> string
  %front = moore.queue.pop_front %queue : <queue<string, 0>> -> string
  // CHECK: moore.queue.sort [[QUEUE]] : <queue<string, 0>>
  moore.queue.sort %queue : <queue<string, 0>>
  // CHECK: moore.queue.delete [[QUEUE]] : <queue<string, 0>>
  moore.queue.delete %queue : <queue<string, 0>>
  return
}

//===----------------------------------------------------------------------===//
// Associative Array Iteration Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @AssocArrayIteration
// CHECK-SAME: ([[ARRAY:%.+]]: !moore.ref<assoc_array<i32, string>>, [[KEY:%.+]]: !moore.ref<string>)
func.func @AssocArrayIteration(%array: !moore.ref<assoc_array<i32, string>>, %key: !moore.ref<string>) {
  // CHECK: moore.assoc.first [[ARRAY]], [[KEY]] : <assoc_array<i32, string>>, <string>
  %found_first = moore.assoc.first %array, %key : <assoc_array<i32, string>>, <string>
  // CHECK: moore.assoc.next [[ARRAY]], [[KEY]] : <assoc_array<i32, string>>, <string>
  %found_next = moore.assoc.next %array, %key : <assoc_array<i32, string>>, <string>
  // CHECK: moore.assoc.last [[ARRAY]], [[KEY]] : <assoc_array<i32, string>>, <string>
  %found_last = moore.assoc.last %array, %key : <assoc_array<i32, string>>, <string>
  // CHECK: moore.assoc.prev [[ARRAY]], [[KEY]] : <assoc_array<i32, string>>, <string>
  %found_prev = moore.assoc.prev %array, %key : <assoc_array<i32, string>>, <string>
  return
}

// Test associative array with different key/value types
// CHECK-LABEL: func.func @AssocArrayWithIntKey
// CHECK-SAME: ([[ARRAY:%.+]]: !moore.ref<assoc_array<string, i64>>, [[KEY:%.+]]: !moore.ref<i64>)
func.func @AssocArrayWithIntKey(%array: !moore.ref<assoc_array<string, i64>>, %key: !moore.ref<i64>) {
  // CHECK: moore.assoc.first [[ARRAY]], [[KEY]] : <assoc_array<string, i64>>, <i64>
  %found_first = moore.assoc.first %array, %key : <assoc_array<string, i64>>, <i64>
  // CHECK: moore.assoc.next [[ARRAY]], [[KEY]] : <assoc_array<string, i64>>, <i64>
  %found_next = moore.assoc.next %array, %key : <assoc_array<string, i64>>, <i64>
  // CHECK: moore.assoc.last [[ARRAY]], [[KEY]] : <assoc_array<string, i64>>, <i64>
  %found_last = moore.assoc.last %array, %key : <assoc_array<string, i64>>, <i64>
  // CHECK: moore.assoc.prev [[ARRAY]], [[KEY]] : <assoc_array<string, i64>>, <i64>
  %found_prev = moore.assoc.prev %array, %key : <assoc_array<string, i64>>, <i64>
  return
}

//===----------------------------------------------------------------------===//
// Associative Array Create Operation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @AssocArrayCreate
func.func @AssocArrayCreate() {
  // CHECK: moore.assoc.create : assoc_array<string, i32>
  %aa1 = moore.assoc.create : !moore.assoc_array<string, i32>
  // CHECK: moore.assoc.create : assoc_array<i32, i32>
  %aa2 = moore.assoc.create : !moore.assoc_array<i32, i32>
  // CHECK: moore.assoc.create : assoc_array<i64, string>
  %aa3 = moore.assoc.create : !moore.assoc_array<i64, string>
  // CHECK: moore.assoc.create : wildcard_assoc_array<i32>
  %aa4 = moore.assoc.create : !moore.wildcard_assoc_array<i32>
  return
}

//===----------------------------------------------------------------------===//
// Associative Array Exists Operation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @AssocArrayExists
// CHECK-SAME: ([[ARRAY:%.+]]: !moore.assoc_array<i32, string>, [[KEY:%.+]]: !moore.string)
func.func @AssocArrayExists(%array: !moore.assoc_array<i32, string>, %key: !moore.string) -> !moore.i1 {
  // CHECK: moore.assoc.exists [[ARRAY]], [[KEY]] : <i32, string>, string
  %exists = moore.assoc.exists %array, %key : !moore.assoc_array<i32, string>, !moore.string
  return %exists : !moore.i1
}

// Test exists with integer key type
// CHECK-LABEL: func.func @AssocArrayExistsIntKey
// CHECK-SAME: ([[ARRAY:%.+]]: !moore.assoc_array<i32, i64>, [[KEY:%.+]]: !moore.i64)
func.func @AssocArrayExistsIntKey(%array: !moore.assoc_array<i32, i64>, %key: !moore.i64) -> !moore.i1 {
  // CHECK: moore.assoc.exists [[ARRAY]], [[KEY]] : <i32, i64>, i64
  %exists = moore.assoc.exists %array, %key : !moore.assoc_array<i32, i64>, !moore.i64
  return %exists : !moore.i1
}
