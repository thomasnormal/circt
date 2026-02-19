// RUN: not circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s
// Test that stochastic queue tasks (§21.6) produce a clear compile-time error.
// These are deprecated Verilog-1364 legacy functions not supported by circt-sim.
module top;
  integer q_id, status, value;

  initial begin
    // CHECK: error: unsupported legacy stochastic queue task '$q_initialize'
    $q_initialize(q_id, 1, 4, status);
  end
endmodule
