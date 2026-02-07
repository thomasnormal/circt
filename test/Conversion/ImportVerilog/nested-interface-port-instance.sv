// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test nested interface port instances: a module port that is a sub-interface
// of another interface (e.g., ParentIf contains ChildIf child).

interface ChildIf(input logic clk);
  logic data;
endinterface

interface ParentIf(input logic clk);
  ChildIf child(.clk(clk));
endinterface

module Consumer(ChildIf child);
endmodule

// CHECK-LABEL: moore.module private @Producer
// CHECK-SAME: in %p : !moore.ref<virtual_interface<@ParentIf>>
// CHECK-SAME: in %parent.child : !moore.ref<virtual_interface<@ChildIf>>
module Producer(ParentIf p);
  // CHECK: moore.virtual_interface.signal_ref {{.*}}[@child] : <@ParentIf> -> <virtual_interface<@ChildIf>>
  // CHECK: moore.instance "u" @Consumer
  Consumer u(.child(p.child));
endmodule

// CHECK-LABEL: moore.module @top
module top;
  logic clk;
  // CHECK: %parent = moore.interface.instance  @ParentIf
  ParentIf parent(.clk(clk));
  // CHECK: moore.virtual_interface.signal_ref {{.*}}[@child] : <@ParentIf> -> <virtual_interface<@ChildIf>>
  // CHECK: moore.instance "p" @Producer
  Producer p(.p(parent));
endmodule
