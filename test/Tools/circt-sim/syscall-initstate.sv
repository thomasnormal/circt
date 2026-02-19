// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $initstate returns 1 during initialization, 0 at runtime.
// Bug: $initstate is stubbed to always return 0.
// IEEE 1800-2017 Section 20.14: $initstate returns 1 when called during
// the initialization phase (i.e., within initial blocks before simulation
// time advances), and 0 afterwards.
module top;
  reg clk = 0;

  initial begin
    // During initialization phase (time 0, no time advance yet)
    // CHECK: initstate_initial=1
    $display("initstate_initial=%0d", $initstate);

    // Advance time — we are now in the runtime phase
    #1;

    // After time has advanced, $initstate should return 0
    // CHECK: initstate_runtime=0
    $display("initstate_runtime=%0d", $initstate);
    $finish;
  end
endmodule
