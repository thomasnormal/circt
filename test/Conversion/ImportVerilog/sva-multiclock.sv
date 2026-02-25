// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module sva_multiclock(input logic clk0, clk1, a, b);
  // CHECK-LABEL: moore.module @sva_multiclock

  property p0;
    @(posedge clk0) a |-> b;
  endproperty
  // CHECK-DAG: [[CLK0_CONV:%.+]] = moore.to_builtin_bool %clk0 : l1
  // CHECK-DAG: verif.clocked_assert {{.*}}, posedge [[CLK0_CONV]] : !ltl.property
  assert property (p0);

  property p1;
    @(posedge clk1) b |-> a;
  endproperty
  // CHECK-DAG: [[CLK1_CONV:%.+]] = moore.to_builtin_bool %clk1 : l1
  // CHECK-DAG: verif.clocked_assert {{.*}}, posedge [[CLK1_CONV]] : !ltl.property
  assert property (p1);
endmodule
