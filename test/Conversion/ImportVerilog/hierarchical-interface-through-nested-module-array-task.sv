// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test hierarchical interface task calls through nested module-instance arrays:
//   outer_array[idx].inner_array[idx].interface_array[idx].task()
// CHECK-LABEL: moore.module @Top()
// CHECK: moore.procedure initial {
// CHECK-COUNT-2: func.call @"IF::ping{{(_[0-9]+)?}}"

interface IF(input bit clk);
  task ping();
    @(posedge clk);
  endtask
endinterface

module Leaf(input bit clk);
  IF ifs[2](clk);
endmodule

module Mid(input bit clk);
  Leaf l[2](clk);
endmodule

module Top #(parameter int MIDX = 1, parameter int LIDX = 0,
             parameter int IIDX = 1);
  bit clk = 0;
  Mid m[2](clk);

  initial begin
    m[MIDX].l[LIDX].ifs[IIDX].ping();
    m[1].l[1 - 1].ifs[1].ping();
  end
endmodule
