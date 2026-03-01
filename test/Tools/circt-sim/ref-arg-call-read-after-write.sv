// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression for issue #50: reads after ref-argument calls must observe
// side effects from the call, not stale pre-call forwarded drives.
module tb;
  task automatic inc(ref int v, input int a);
    $display("before: v=%0d", v);
    v += a;
    $display("after: v=%0d", v);
  endtask

  int x;
  initial begin
    x = 10;
    $display("x before call: %0d", x);
    inc(x, 5);
    $display("x after call: %0d", x);
    if (x == 15) $display("PASS");
    else         $display("FAIL x=%0d (expect 15)", x);

    // CHECK: x before call: 10
    // CHECK: before: v=10
    // CHECK: after: v=15
    // CHECK: x after call: 15
    // CHECK: PASS
    // CHECK-NOT: FAIL x=

    $finish;
  end
endmodule
