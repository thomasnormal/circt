// RUN: circt-verilog --ir-moore %s | FileCheck %s

interface ChildIf(input logic clk);
  logic data;
endinterface

interface ParentIf(input logic clk);
  ChildIf child(.clk(clk));
endinterface

module Consumer(ChildIf child);
endmodule

module top;
  logic clk;
  ParentIf parent(.clk(clk));
  Consumer u(.child(parent.child));
endmodule

// CHECK: moore.interface @ChildIf
// CHECK: moore.interface @ParentIf
// CHECK: moore.interface.signal @child : !moore.virtual_interface<@ChildIf> {interface_instance}
// CHECK: moore.module @top
// CHECK: moore.interface.instance @ParentIf
// CHECK: moore.virtual_interface.signal_ref
