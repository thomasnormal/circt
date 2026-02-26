// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test hierarchical interface signal access through nested module-instance arrays:
//   outer_array[idx].inner_array[idx].interface_array[idx].signal
// CHECK-LABEL: moore.module @Top()
// CHECK: moore.procedure initial {
// CHECK-COUNT-2: moore.virtual_interface.signal_ref

interface IF;
  logic v;
endinterface

module Leaf;
  IF ifs[2]();
endmodule

module Mid;
  Leaf l[2]();
endmodule

module Top #(parameter int MIDX = 1, parameter int LIDX = 0,
             parameter int IIDX = 1);
  Mid m[2]();
  logic x;

  initial begin
    m[MIDX].l[LIDX].ifs[IIDX].v = 1'b1;
    x = m[1].l[1 - 1].ifs[1].v;
  end
endmodule
