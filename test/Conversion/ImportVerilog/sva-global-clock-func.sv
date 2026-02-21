// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s --check-prefix=CHECK-MOORE
// REQUIRES: slang

module SvaGlobalClockFunc(input logic clk, a, b);
  global clocking @(posedge clk); endclocking

  // $global_clock in assertion timing control should resolve to the declared
  // global clocking event and produce a clocked assertion.
  // CHECK-LABEL: moore.module @SvaGlobalClockFunc
  // CHECK: ltl.clock
  // CHECK-NEXT: verif.assert
  assert property (@($global_clock) (a |-> b));

  // CHECK-MOORE-LABEL: moore.module @SvaGlobalClockFunc
  // CHECK-MOORE: verif.clocked_assert
endmodule
