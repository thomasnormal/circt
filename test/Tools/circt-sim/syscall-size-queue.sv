// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $size() returns actual queue size at runtime
module top;
  int q[$];
  initial begin
    q.push_back(10);
    q.push_back(20);
    q.push_back(30);
    // CHECK: size=3
    $display("size=%0d", $size(q));
    $finish;
  end
endmodule
