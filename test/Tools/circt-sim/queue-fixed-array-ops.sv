// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test queue/array operations on fixed-size arrays (UnpackedArrayType).
// Commit c52eee8a9 extended QueueReverse, QueueShuffle, QueueRSort,
// QueueReduce, and QueueUniqueIndex conversion patterns to handle
// fixed-size arrays in addition to queues. For fixed-size arrays, the
// adapted value is !llhd.ref<!hw.array<N x T>> rather than a queue
// struct pointer.

module top;
  int arr[5];
  int tmp[5];
  int pair[2];

  initial begin
    arr[0] = 5; arr[1] = 3; arr[2] = 1; arr[3] = 4; arr[4] = 2;

    // ---- Test reverse on fixed-size array ----
    tmp = arr;
    tmp.reverse();
    // CHECK: reverse: 2 4 1 3 5
    $display("reverse: %0d %0d %0d %0d %0d",
             tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]);

    // ---- Test double reverse returns to original ----
    tmp.reverse();
    // CHECK: double-reverse: 5 3 1 4 2
    $display("double-reverse: %0d %0d %0d %0d %0d",
             tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]);

    // ---- Test rsort on fixed-size array (descending order) ----
    tmp = arr;
    tmp.rsort();
    // CHECK: rsort: 5 4 3 2 1
    $display("rsort: %0d %0d %0d %0d %0d",
             tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]);

    // ---- Test shuffle on fixed-size array ----
    // Since shuffle is random, verify all elements are preserved by
    // sorting after shuffle.
    tmp = arr;
    tmp.shuffle();
    tmp.rsort();
    // CHECK: shuffle-rsort: 5 4 3 2 1
    $display("shuffle-rsort: %0d %0d %0d %0d %0d",
             tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]);

    // ---- Test rsort is idempotent ----
    tmp.rsort();
    // CHECK: rsort-idempotent: 5 4 3 2 1
    $display("rsort-idempotent: %0d %0d %0d %0d %0d",
             tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]);

    // ---- Test reverse on 2-element array ----
    pair[0] = 10;
    pair[1] = 20;
    pair.reverse();
    // CHECK: pair-reverse: 20 10
    $display("pair-reverse: %0d %0d", pair[0], pair[1]);

    // ---- Test rsort on 2-element array ----
    pair.rsort();
    // CHECK: pair-rsort: 20 10
    $display("pair-rsort: %0d %0d", pair[0], pair[1]);

    // ---- Test reverse on already-reversed (restores original) ----
    pair.reverse();
    // CHECK: pair-reverse-again: 10 20
    $display("pair-reverse-again: %0d %0d", pair[0], pair[1]);

    // ---- Test rsort on ascending data ----
    pair[0] = 10;
    pair[1] = 20;
    pair.rsort();
    // CHECK: pair-rsort-asc: 20 10
    $display("pair-rsort-asc: %0d %0d", pair[0], pair[1]);

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule
