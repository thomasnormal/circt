// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test transitive nested interface forwarding across module ports:
//   TopIf.top.master.write passed through Mid(MasterIf) to Leaf(WriteIf).

interface WriteIf;
  logic data;
endinterface

interface MasterIf;
  WriteIf write();
endinterface

interface TopIf;
  MasterIf master();
endinterface

module Leaf(WriteIf w);
endmodule

// CHECK-LABEL: moore.module private @Mid
module Mid(MasterIf m);
  // CHECK: moore.virtual_interface.signal_ref {{.*}}[@write] : <@MasterIf> -> <virtual_interface<@WriteIf>>
  // CHECK: moore.instance "u_leaf" @Leaf
  Leaf u_leaf(.w(m.write));
endmodule

// CHECK-LABEL: moore.module @Top
module Top;
  // CHECK: %t = moore.interface.instance  @TopIf
  TopIf t();
  // CHECK: moore.virtual_interface.signal_ref {{.*}}[@master] : <@TopIf> -> <virtual_interface<@MasterIf>>
  // CHECK: moore.virtual_interface.signal_ref {{.*}}[@write] : <@MasterIf> -> <virtual_interface<@WriteIf>>
  // CHECK: moore.instance "u_mid" @Mid
  Mid u_mid(.m(t.master));
endmodule
