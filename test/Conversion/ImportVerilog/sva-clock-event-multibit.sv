// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaClockEventMultibit(input logic [1:0] e, input logic a);
  // Event controls should accept integral truthy expressions in clock slots.
  // CHECK-LABEL: moore.module @SvaClockEventMultibit
  // CHECK: moore.bool_cast
  // CHECK: verif.clocked_assert {{.*}}, edge {{.*}} : i1
  assert property (@(e) a);
endmodule
