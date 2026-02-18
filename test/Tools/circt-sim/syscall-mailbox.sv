// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test mailbox: new, put, get, try_put, try_get, num, peek, try_peek
module top;
  mailbox #(int) mb;
  int val;
  int ok;

  initial begin
    mb = new(4);  // bounded mailbox with size 4

    // num should be 0
    // CHECK: num_empty=0
    $display("num_empty=%0d", mb.num());

    // put and get
    mb.put(42);
    mb.put(99);
    // CHECK: num_after_put=2
    $display("num_after_put=%0d", mb.num());

    mb.get(val);
    // CHECK: get_first=42
    $display("get_first=%0d", val);

    // peek (doesn't remove)
    mb.peek(val);
    // CHECK: peek=99
    $display("peek=%0d", val);
    // CHECK: num_after_peek=1
    $display("num_after_peek=%0d", mb.num());

    // try_get
    ok = mb.try_get(val);
    // CHECK: try_get_ok=1
    $display("try_get_ok=%0d", ok);
    // CHECK: try_get_val=99
    $display("try_get_val=%0d", val);

    // try_get on empty
    ok = mb.try_get(val);
    // CHECK: try_get_empty=0
    $display("try_get_empty=%0d", ok);

    // try_put
    ok = mb.try_put(77);
    // CHECK: try_put_ok=1
    $display("try_put_ok=%0d", ok);

    $finish;
  end
endmodule
