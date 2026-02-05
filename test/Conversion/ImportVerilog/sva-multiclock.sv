// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module sva_multiclock(input logic clk0, clk1, a, b);
  // CHECK-LABEL: moore.module @sva_multiclock

  property p0;
    @(posedge clk0) a |-> b;
  endproperty
  // CHECK-DAG: [[CLK0_VAR:%.+]] = moore.variable name "clk0" : <l1>
  // CHECK-DAG: [[CLK0_READ:%.+]] = moore.read [[CLK0_VAR]] : <l1>
  // CHECK-DAG: [[CLK0_CONV:%.+]] = moore.to_builtin_bool [[CLK0_READ]] : l1
  // CHECK-DAG: ltl.clock {{.*}}, posedge [[CLK0_CONV]] : !ltl.property
  assert property (p0);

  property p1;
    @(posedge clk1) b |-> a;
  endproperty
  // CHECK-DAG: [[CLK1_VAR:%.+]] = moore.variable name "clk1" : <l1>
  // CHECK-DAG: [[CLK1_READ:%.+]] = moore.read [[CLK1_VAR]] : <l1>
  // CHECK-DAG: [[CLK1_CONV:%.+]] = moore.to_builtin_bool [[CLK1_READ]] : l1
  // CHECK-DAG: ltl.clock {{.*}}, posedge [[CLK1_CONV]] : !ltl.property
  assert property (p1);
endmodule
