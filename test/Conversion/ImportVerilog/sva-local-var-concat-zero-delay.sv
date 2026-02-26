// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module sva_local_var_concat_zero_delay(
    input logic clk, valid,
    input logic [7:0] in,
    output logic [7:0] out);
  property p;
    int x;
    @(posedge clk) (valid, x = in) ##0 (valid && out == x[7:0]);
  endproperty

  // CHECK-LABEL: moore.module @sva_local_var_concat_zero_delay
  // CHECK: [[X_I32:%.*]] = moore.logic_to_int
  // CHECK-NOT: moore.past [[X_I32]] delay 1
  // CHECK: [[X_I8:%.*]] = moore.extract [[X_I32]] from 0 : i32 -> i8
  // CHECK: [[CMP:%.*]] = moore.eq
  // CHECK: [[OVERLAP:%.*]] = ltl.and
  // CHECK: verif.clocked_assert [[OVERLAP]], posedge
  assert property (p);
endmodule
