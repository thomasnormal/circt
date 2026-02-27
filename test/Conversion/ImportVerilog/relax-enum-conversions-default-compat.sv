// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Default behavior should match mainstream simulator compatibility for
// implicit enum conversions, with strict mode still available via
// --relax-enum-conversions=false.

// CHECK-LABEL: moore.module @top
// CHECK: %[[ONE:.+]] = moore.constant 1 : i32
// CHECK: moore.blocking_assign %i, %[[ONE]] : i32
// CHECK: %[[INTVAL:.+]] = moore.read %i : <i32>
// CHECK: %[[ENUMINT:.+]] = moore.trunc %[[INTVAL]] : i32 -> i2
// CHECK: %[[ENUMLOGIC:.+]] = moore.int_to_logic %[[ENUMINT]] : i2
// CHECK: moore.blocking_assign %s, %[[ENUMLOGIC]] : l2
module top;
  typedef enum logic [1:0] {
    S0 = 2'b00,
    S1 = 2'b01
  } state_t;

  state_t s;
  int i;

  initial begin
    i = 1;
    s = i;
  end
endmodule
