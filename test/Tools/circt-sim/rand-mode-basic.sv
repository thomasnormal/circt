// RUN: circt-verilog %s --ir-hw -o %t.mlir && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// REQUIRES: slang

// Test basic rand_mode state tracking: default=1, disable->0, re-enable->1
class MyClass;
  rand int x;

  function new();
    x = 0;
  endfunction
endclass

module top;
  initial begin
    MyClass obj = new();
    int ret1, ret2, ret3;

    // Default rand_mode should be 1 (enabled)
    ret1 = obj.x.rand_mode();
    $display("ret1=%0d", ret1);

    // Disable rand_mode
    obj.x.rand_mode(0);
    ret2 = obj.x.rand_mode();
    $display("ret2=%0d", ret2);

    // Re-enable rand_mode
    obj.x.rand_mode(1);
    ret3 = obj.x.rand_mode();
    $display("ret3=%0d", ret3);

    // CHECK: ret1=1
    // CHECK: ret2=0
    // CHECK: ret3=1
    $finish;
  end
endmodule
