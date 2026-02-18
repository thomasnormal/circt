// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $finish with various verbosity levels and $stop
module top;
  initial begin
    // CHECK: before_finish
    $display("before_finish");
    // $finish(0) — silent finish
    $finish(0);
    // This should NOT be reached
    // CHECK-NOT: after_finish
    $display("after_finish");
  end
endmodule
