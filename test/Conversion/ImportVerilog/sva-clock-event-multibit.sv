// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaClockEventMultibit(input logic [1:0] e, input logic a);
  // Event controls should accept integral truthy expressions in clock slots.
  // CHECK-LABEL: moore.module @SvaClockEventMultibit
  // CHECK: moore.bool_cast
  // CHECK: ltl.clock
  // CHECK: verif.assert
  assert property (@(e) a);
endmodule
