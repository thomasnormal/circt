// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// IEEE 1800-2017 Section 7.12.2 "Array locator methods"
// Test array locator methods: find, find_index, find_first, find_first_index,
// find_last, find_last_index

// CHECK-LABEL: func.func private @ArrayLocatorMethods
function void ArrayLocatorMethods(int q[$], int arr[5]);
  int result[$];
  int idx_result[$];

  // CHECK: moore.array.locator all, elements
  // CHECK-SAME: queue<i32, 0>
  // CHECK: ^bb0(%[[ARG:.*]]: !moore.i32, %[[IDX:.*]]: !moore.i32):
  // CHECK:   %[[CONST:.*]] = moore.constant 2 : i32
  // CHECK:   %[[CMP:.*]] = moore.sgt %[[ARG]], %[[CONST]] : i32
  // CHECK:   moore.array.locator.yield %[[CMP]] : i1
  result = q.find(x) with (x > 2);

  // CHECK: moore.array.locator all, indices
  // CHECK-SAME: queue<i32, 0>
  // CHECK: ^bb0(%[[ARG:.*]]: !moore.i32, %[[IDX:.*]]: !moore.i32):
  // CHECK:   moore.array.locator.yield
  idx_result = q.find_index(x) with (x > 2);

  // CHECK: moore.array.locator first, elements
  // CHECK-SAME: queue<i32, 0>
  // CHECK: ^bb0(%[[ARG:.*]]: !moore.i32, %[[IDX:.*]]: !moore.i32):
  // CHECK:   moore.array.locator.yield
  result = q.find_first(x) with (x > 2);

  // CHECK: moore.array.locator first, indices
  // CHECK-SAME: queue<i32, 0>
  // CHECK: ^bb0(%[[ARG:.*]]: !moore.i32, %[[IDX:.*]]: !moore.i32):
  // CHECK:   moore.array.locator.yield
  idx_result = q.find_first_index(x) with (x > 2);

  // CHECK: moore.array.locator last, elements
  // CHECK-SAME: queue<i32, 0>
  // CHECK: ^bb0(%[[ARG:.*]]: !moore.i32, %[[IDX:.*]]: !moore.i32):
  // CHECK:   moore.array.locator.yield
  result = q.find_last(x) with (x > 2);

  // CHECK: moore.array.locator last, indices
  // CHECK-SAME: queue<i32, 0>
  // CHECK: ^bb0(%[[ARG:.*]]: !moore.i32, %[[IDX:.*]]: !moore.i32):
  // CHECK:   moore.array.locator.yield
  idx_result = q.find_last_index(x) with (x > 2);

  // Test with fixed-size array
  // CHECK: moore.array.locator all, elements
  // CHECK-SAME: uarray<5 x i32>
  result = arr.find(x) with (x >= 30);

  // CHECK: moore.array.locator first, indices
  // CHECK-SAME: uarray<5 x i32>
  idx_result = arr.find_first_index(x) with (x == 30);

endfunction

// Test item.index support in array locator methods
// IEEE 1800-2017 Section 7.12.1 "Array manipulation methods"
// CHECK-LABEL: func.func private @ArrayLocatorWithIndex
function void ArrayLocatorWithIndex(int q[$]);
  int result[$];

  // item.index should return the current index during iteration
  // CHECK: moore.array.locator all, elements
  // CHECK: ^bb0(%[[ITEM:.*]]: !moore.i32, %[[IDX:.*]]: !moore.i32):
  // CHECK:   moore.eq %[[ITEM]], %[[IDX]] : i32
  // CHECK:   moore.array.locator.yield
  result = q.find(item) with (item == item.index);

endfunction
