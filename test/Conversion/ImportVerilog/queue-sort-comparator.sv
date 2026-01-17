// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test queue sort with custom comparator using "with" clause
// IEEE 1800-2017 Section 7.12.3 "Array ordering methods"

module queue_sort_comparator;

  // Test basic sort operations directly in procedural code
  // CHECK-LABEL: moore.procedure always_comb
  always_comb begin
    int q[$];
    q = '{3, 1, 4, 1, 5};

    // Simple sort (no comparator)
    // CHECK: moore.queue.sort %{{.*}} : <queue<i32, 0>>
    q.sort();

    // Simple rsort (no comparator)
    // CHECK: moore.queue.rsort %{{.*}} : <queue<i32, 0>>
    q.rsort();

    // Sort with key expression
    // CHECK: moore.queue.sort.with %{{.*}} : <queue<i32, 0>>
    // CHECK:   moore.queue.sort.key.yield
    q.sort() with (item % 10);

    // Rsort with key expression
    // CHECK: moore.queue.rsort.with %{{.*}} : <queue<i32, 0>>
    // CHECK:   moore.queue.sort.key.yield
    q.rsort() with (item % 10);

    // Shuffle
    // CHECK: moore.queue.shuffle %{{.*}} : <queue<i32, 0>>
    q.shuffle();
  end

endmodule
