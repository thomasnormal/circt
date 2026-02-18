// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test queue methods: push_back, push_front, pop_back, pop_front, size, delete
module top;
  int q[$];
  int val;

  initial begin
    // Initially empty
    // CHECK: size_init=0
    $display("size_init=%0d", q.size());

    // push_back
    q.push_back(10);
    q.push_back(20);
    q.push_back(30);
    // CHECK: size_after_push=3
    $display("size_after_push=%0d", q.size());

    // Access by index
    // CHECK: q0=10
    $display("q0=%0d", q[0]);
    // CHECK: q1=20
    $display("q1=%0d", q[1]);
    // CHECK: q2=30
    $display("q2=%0d", q[2]);

    // pop_front removes and returns first element
    val = q.pop_front();
    // CHECK: pop_front=10
    $display("pop_front=%0d", val);
    // CHECK: size_after_pop_front=2
    $display("size_after_pop_front=%0d", q.size());

    // pop_back removes and returns last element
    val = q.pop_back();
    // CHECK: pop_back=30
    $display("pop_back=%0d", val);
    // CHECK: size_after_pop_back=1
    $display("size_after_pop_back=%0d", q.size());

    // push_front adds to front
    q.push_front(5);
    // CHECK: after_push_front_0=5
    $display("after_push_front_0=%0d", q[0]);
    // CHECK: after_push_front_1=20
    $display("after_push_front_1=%0d", q[1]);

    // delete clears the queue
    q.delete();
    // CHECK: size_after_delete=0
    $display("size_after_delete=%0d", q.size());

    $finish;
  end
endmodule
