// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $initstate semantics in mixed process topologies:
// - Returns 1 at time-0 before any time advance
// - Returns 0 after time has advanced (even in the same initial block)
module top;
  initial begin
    // At time 0: initstate should be 1
    // CHECK: init_t0=1
    $display("init_t0=%0d", $initstate);

    // Advance time
    #1;

    // After time advance: initstate should be 0
    // CHECK: init_t1=0
    $display("init_t1=%0d", $initstate);

    $finish;
  end
endmodule
