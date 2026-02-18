// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test enum methods: first, last, next, prev, num, name
module top;
  typedef enum {RED=0, GREEN=1, BLUE=2, YELLOW=3} color_t;
  color_t c;

  initial begin
    c = RED;

    // first
    // CHECK: first=0
    $display("first=%0d", c.first());

    // last
    // CHECK: last=3
    $display("last=%0d", c.last());

    // next
    c = GREEN;
    // CHECK: next=2
    $display("next=%0d", c.next());

    // prev
    c = BLUE;
    // CHECK: prev=1
    $display("prev=%0d", c.prev());

    // num
    // CHECK: num=4
    $display("num=%0d", c.num());

    // name
    c = BLUE;
    // CHECK: name=BLUE
    $display("name=%s", c.name());

    $finish;
  end
endmodule
