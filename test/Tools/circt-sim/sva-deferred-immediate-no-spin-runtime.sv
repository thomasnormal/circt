// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=1000000 2>&1 | FileCheck %s

// Regression: module-scope deferred immediate assertions used to lower to
// tight self-loops and trip the per-activation step-limit guard.
// Keep this as a non-failing condition to avoid flooding stderr while still
// checking for the loop regression.

// CHECK-NOT: exceeded per-activation step limit
// CHECK: [circt-sim] Simulation completed

module top;
  logic a = 1'b1;
  assume #0 (a);
  assume final (a);
endmodule
