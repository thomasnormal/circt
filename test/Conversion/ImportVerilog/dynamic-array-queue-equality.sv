// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module DynamicArrayQueueEquality(output logic dae_eq_o, output logic dae_ne_o,
                                 output logic q_eq_o, output logic q_ne_o);
  int a[];
  int b[];
  int q0[$];
  int q1[$];

  always_comb begin
    dae_eq_o = (a == b);
    dae_ne_o = (a != b);
    q_eq_o = (q0 == q1);
    q_ne_o = (q0 != q1);
  end

  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.dyn_extract
  // CHECK-DAG: moore.eq
  // CHECK-DAG: moore.and
  // CHECK-DAG: moore.not
  // CHECK-DAG: moore.blocking_assign
endmodule
