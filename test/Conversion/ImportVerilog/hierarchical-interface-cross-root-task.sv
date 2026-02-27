// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test hierarchical interface task calls across sibling top-level roots and
// verify functional behavior (state updates observed after calls):
//   caller -> topa.ifc.ping(11) updates topa.ifc.state
//   caller -> topb.ifc.ping(22) updates topb.ifc.state
// CHECK-LABEL: moore.module @caller(
// CHECK-SAME: in %topa.ifc : !moore.ref<virtual_interface<@IF>>
// CHECK-SAME: in %topb.ifc : !moore.ref<virtual_interface<@IF>>
// CHECK-DAG: moore.constant 11 : i32
// CHECK-DAG: moore.constant 22 : i32
// CHECK: moore.procedure initial {
// CHECK: moore.read %topa.ifc : <virtual_interface<@IF>>
// CHECK-COUNT-2: func.call @"IF::ping{{(_[0-9]+)?}}"
// CHECK: moore.virtual_interface.signal_ref
// CHECK: moore.case_ne
// CHECK: moore.builtin.severity fatal
// CHECK: moore.virtual_interface.signal_ref
// CHECK: moore.case_ne
// CHECK: moore.builtin.severity fatal

interface IF;
  int state;

  task ping(input int v);
    state = v;
  endtask
endinterface

module topa;
  IF ifc();
endmodule

module topb;
  IF ifc();
endmodule

module caller;
  initial begin
    topa.ifc.ping(11);
    topb.ifc.ping(22);
    if (topa.ifc.state !== 11)
      $fatal(1, "topa.ifc.state mismatch");
    if (topb.ifc.state !== 22)
      $fatal(1, "topb.ifc.state mismatch");
    $finish;
  end
endmodule
