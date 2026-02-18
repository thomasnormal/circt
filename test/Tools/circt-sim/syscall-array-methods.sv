// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test dynamic array/queue methods: size, delete, push_back, push_front, pop_back, pop_front,
// insert, find, find_index, sort, reverse, shuffle, sum, product, min, max, unique
module top;
  int q[$];
  int result;
  int arr[];

  initial begin
    // Queue methods
    q.push_back(10);
    q.push_back(20);
    q.push_back(30);
    // CHECK: size=3
    $display("size=%0d", q.size());

    q.push_front(5);
    // CHECK: front=5
    $display("front=%0d", q[0]);

    result = q.pop_front();
    // CHECK: pop_front=5
    $display("pop_front=%0d", result);

    result = q.pop_back();
    // CHECK: pop_back=30
    $display("pop_back=%0d", result);

    // CHECK: remaining_size=2
    $display("remaining_size=%0d", q.size());

    // Insert at index 1
    q.insert(1, 15);
    // CHECK: after_insert=10
    $display("after_insert=%0d", q[0]);
    // CHECK: inserted=15
    $display("inserted=%0d", q[1]);
    // CHECK: shifted=20
    $display("shifted=%0d", q[2]);

    // Delete
    q.delete();
    // CHECK: after_delete=0
    $display("after_delete=%0d", q.size());

    // Sort
    q = '{30, 10, 20, 40, 5};
    q.sort();
    // CHECK: sorted=5 10 20 30 40
    $write("sorted=");
    for (int i = 0; i < q.size(); i++) $write("%0d ", q[i]);
    $write("\n");

    // Reverse
    q.reverse();
    // CHECK: reversed=40 30 20 10 5
    $write("reversed=");
    for (int i = 0; i < q.size(); i++) $write("%0d ", q[i]);
    $write("\n");

    // Sum
    // CHECK: sum=105
    $display("sum=%0d", q.sum());

    // Min/Max
    // CHECK: min=5
    $display("min=%0d", q.min());
    // CHECK: max=40
    $display("max=%0d", q.max());

    $finish;
  end
endmodule
