// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $initstate returns 1 inside an initial block.
// Bug: $initstate is stubbed to always return 0.
// IEEE 1800-2017 Section 20.14: $initstate returns 1 when called during
// the initialization phase (i.e., within initial blocks before simulation
// time advances).
module top;
  initial begin
    // CHECK: initstate=1
    $display("initstate=%0d", $initstate);
    $finish;
  end
endmodule
