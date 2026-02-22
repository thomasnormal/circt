// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module DynamicArrayQueueCaseEquality(output logic dae_ceq_o, output logic dae_cne_o,
                                     output logic q_ceq_o, output logic q_cne_o);
  int a[];
  int b[];
  int q0[$];
  int q1[$];

  always_comb begin
    dae_ceq_o = (a === b);
    dae_cne_o = (a !== b);
    q_ceq_o = (q0 === q1);
    q_cne_o = (q0 !== q1);
  end

  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.dyn_extract
  // CHECK-DAG: moore.case_eq
  // CHECK-DAG: moore.and
  // CHECK-DAG: moore.not
  // CHECK-DAG: moore.blocking_assign
endmodule
