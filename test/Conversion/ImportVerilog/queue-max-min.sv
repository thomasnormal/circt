// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Array Locator Methods (IEEE 1800-2017 Section 7.12.3)
// Tests for queue max() and min() methods
//===----------------------------------------------------------------------===//

/// Test queue max() method - returns queue with maximum value(s)
// CHECK-LABEL: moore.module @QueueMaxTest() {
module QueueMaxTest;
    int q[$];
    int max_vals[$];

    initial begin
        q.push_back(10);
        q.push_back(30);
        q.push_back(20);
        // CHECK: [[Q_READ:%.+]] = moore.read %q
        // CHECK: [[MAX:%.+]] = moore.queue.max [[Q_READ]] : !moore.queue<i32, 0> -> !moore.queue<i32, 0>
        max_vals = q.max();
    end
endmodule

/// Test queue min() method - returns queue with minimum value(s)
// CHECK-LABEL: moore.module @QueueMinTest() {
module QueueMinTest;
    int q[$];
    int min_vals[$];

    initial begin
        q.push_back(10);
        q.push_back(5);
        q.push_back(20);
        // CHECK: [[Q_READ:%.+]] = moore.read %q
        // CHECK: [[MIN:%.+]] = moore.queue.min [[Q_READ]] : !moore.queue<i32, 0> -> !moore.queue<i32, 0>
        min_vals = q.min();
    end
endmodule

/// Test fixed array max() and min() methods
// CHECK-LABEL: moore.module @FixedArrayMaxMinTest() {
module FixedArrayMaxMinTest;
    int arr[4];
    int max_result[$];
    int min_result[$];

    initial begin
        arr[0] = 1;
        arr[1] = 4;
        arr[2] = 2;
        arr[3] = 3;
        // CHECK: [[ARR_READ:%.+]] = moore.read %arr
        // CHECK: moore.queue.max [[ARR_READ]] : !moore.uarray<4 x i32> -> !moore.queue<i32, 0>
        max_result = arr.max();
        // CHECK: [[ARR_READ2:%.+]] = moore.read %arr
        // CHECK: moore.queue.min [[ARR_READ2]] : !moore.uarray<4 x i32> -> !moore.queue<i32, 0>
        min_result = arr.min();
    end
endmodule
