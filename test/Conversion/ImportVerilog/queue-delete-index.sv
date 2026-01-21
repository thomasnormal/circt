// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Queue delete(index) Tests
//===----------------------------------------------------------------------===//

/// Test queue delete(index) method - removes element at specific index
// CHECK-LABEL: moore.module @QueueDeleteIndexTest() {
module QueueDeleteIndexTest;
    int q[$];
    int idx;

    initial begin
        q.push_back(10);
        q.push_back(20);
        q.push_back(30);
        q.push_back(40);

        // Delete element at index 1 (value 20)
        // CHECK: [[IDX1:%.+]] = moore.constant 1 : i32
        // CHECK: moore.queue.delete %q{{\[}}[[IDX1]]{{\]}} : <queue<i32, 0>>, !moore.i32
        q.delete(1);

        // Delete element at index 0 (first element)
        // CHECK: [[IDX0:%.+]] = moore.constant 0 : i32
        // CHECK: moore.queue.delete %q{{\[}}[[IDX0]]{{\]}} : <queue<i32, 0>>, !moore.i32
        q.delete(0);

        // Delete using variable index
        idx = 1;
        // CHECK: [[IDXVAR:%.+]] = moore.read %idx : <i32>
        // CHECK: moore.queue.delete %q{{\[}}[[IDXVAR]]{{\]}} : <queue<i32, 0>>, !moore.i32
        q.delete(idx);
    end
endmodule

/// Test delete(index) vs delete() - both forms should be distinct
// CHECK-LABEL: moore.module @QueueDeleteBothFormsTest() {
module QueueDeleteBothFormsTest;
    int q[$];

    initial begin
        q.push_back(1);
        q.push_back(2);
        q.push_back(3);

        // Delete specific index
        // CHECK: [[IDX:%.+]] = moore.constant 0 : i32
        // CHECK: moore.queue.delete %q{{\[}}[[IDX]]{{\]}} : <queue<i32, 0>>, !moore.i32
        q.delete(0);

        // Delete all elements (no index)
        // CHECK: moore.queue.delete %q : <queue<i32, 0>>
        q.delete();
    end
endmodule

/// Test delete(index) with different queue element types
// CHECK-LABEL: moore.module @QueueDeleteIndexTypesTest() {
module QueueDeleteIndexTypesTest;
    byte byte_q[$];
    longint long_q[$];
    logic [31:0] logic_q[$];

    initial begin
        byte_q.push_back(8'hAA);
        byte_q.push_back(8'hBB);
        // CHECK: moore.queue.delete %byte_q{{\[}}{{%.+}}{{\]}} : <queue<i8, 0>>, !moore.i32
        byte_q.delete(0);

        long_q.push_back(64'd100);
        long_q.push_back(64'd200);
        // CHECK: moore.queue.delete %long_q{{\[}}{{%.+}}{{\]}} : <queue<i64, 0>>, !moore.i32
        long_q.delete(1);

        logic_q.push_back(32'hDEAD);
        logic_q.push_back(32'hBEEF);
        // CHECK: moore.queue.delete %logic_q{{\[}}{{%.+}}{{\]}} : <queue<l32, 0>>, !moore.i32
        logic_q.delete(0);
    end
endmodule

/// Test delete(index) in a loop pattern (common UVM usage)
// CHECK-LABEL: moore.module @QueueDeleteInLoopTest() {
module QueueDeleteInLoopTest;
    int q[$];
    int i;

    initial begin
        // Populate queue
        for (i = 0; i < 5; i++)
            q.push_back(i * 10);

        // Delete from end backwards (common pattern to avoid index shifting issues)
        for (i = 4; i >= 0; i--) begin
            // CHECK: moore.queue.delete %q
            q.delete(i);
        end
    end
endmodule

/// Test delete(index) with $ (last element index)
// CHECK-LABEL: moore.module @QueueDeleteLastTest() {
module QueueDeleteLastTest;
    int q[$];

    initial begin
        q.push_back(1);
        q.push_back(2);
        q.push_back(3);

        // Delete last element using $ which is (size - 1)
        // CHECK: moore.array.size
        // CHECK: moore.sub
        // CHECK: moore.queue.delete %q
        q.delete(q.size() - 1);
    end
endmodule
