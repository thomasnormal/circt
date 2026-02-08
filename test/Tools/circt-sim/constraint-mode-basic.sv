// RUN: circt-verilog %s --ir-hw -o %t.mlir && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// REQUIRES: slang

// Test basic constraint_mode state tracking
class MyClass;
  rand int x;
  constraint c1 { x > 0; }
  constraint c2 { x < 100; }

  function new();
    x = 0;
  endfunction
endclass

module top;
  initial begin
    MyClass obj = new();
    int ret1, ret2, ret3, ret4;

    // Default constraint_mode should be 1 (enabled)
    ret1 = obj.c1.constraint_mode();
    $display("c1_default=%0d", ret1);

    // Disable c1
    obj.c1.constraint_mode(0);
    ret2 = obj.c1.constraint_mode();
    $display("c1_disabled=%0d", ret2);

    // c2 should still be enabled
    ret3 = obj.c2.constraint_mode();
    $display("c2_still=%0d", ret3);

    // Re-enable c1
    obj.c1.constraint_mode(1);
    ret4 = obj.c1.constraint_mode();
    $display("c1_reenabled=%0d", ret4);

    // CHECK: c1_default=1
    // CHECK: c1_disabled=0
    // CHECK: c2_still=1
    // CHECK: c1_reenabled=1
    $finish;
  end
endmodule
