// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

interface Child;
  logic awvalid;
endinterface

interface Parent;
  Child child();
endinterface

module top;
  Parent p();
  logic x;
  assign p.child.awvalid = x;
endmodule

// CHECK-LABEL: moore.interface @Child
// CHECK: moore.interface.signal @awvalid : !moore.l1
// CHECK-LABEL: moore.interface @Parent
// CHECK: moore.interface.signal @child : !moore.virtual_interface<@Child>
// CHECK-LABEL: moore.module @top
// CHECK: %[[P:.*]] = moore.interface.instance @Parent
// CHECK: %[[PVAL:.*]] = moore.read %[[P]]
// CHECK: %[[CHILD_REF:.*]] = moore.virtual_interface.signal_ref %[[PVAL]][@child]
// CHECK: %[[CHILD_VAL:.*]] = moore.read %[[CHILD_REF]]
// CHECK: %[[AW_REF:.*]] = moore.virtual_interface.signal_ref %[[CHILD_VAL]][@awvalid]
// CHECK: %[[X:.*]] = moore.read
// CHECK: moore.assign %[[AW_REF]], %[[X]]
