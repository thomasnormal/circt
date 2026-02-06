// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

module top;
  int q[$];
  int r[$];

  initial begin
    q.push_back(42);
    q.push_back(7);
    q.push_back(42);
    q.push_back(99);
    q.push_back(7);
    q.push_back(3);
    q.push_back(99);

    // CHECK: size before = 7
    $display("size before = %0d", q.size());

    r = q.unique();

    // CHECK: unique size = 4
    $display("unique size = %0d", r.size());

    // Sort unique result to verify all distinct values present
    r.sort();

    // CHECK: r[0] = 3
    $display("r[0] = %0d", r[0]);
    // CHECK: r[1] = 7
    $display("r[1] = %0d", r[1]);
    // CHECK: r[2] = 42
    $display("r[2] = %0d", r[2]);
    // CHECK: r[3] = 99
    $display("r[3] = %0d", r[3]);

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule
