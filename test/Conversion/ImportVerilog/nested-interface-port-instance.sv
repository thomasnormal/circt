// RUN: circt-verilog --ir-moore %s | FileCheck %s

interface ChildIf(input logic clk);
  logic data;
endinterface

interface ParentIf(input logic clk);
  ChildIf child(.clk(clk));
endinterface

module Consumer(ChildIf child);
endmodule

module Producer(ParentIf p);
  Consumer u(.child(p.child));
endmodule

module top;
  logic clk;
  ParentIf parent(.clk(clk));
  Producer p(.p(parent));
endmodule

// CHECK: moore.interface @ChildIf
// CHECK: moore.interface @ParentIf
// CHECK: moore.module private @Producer
// CHECK: moore.virtual_interface.signal_ref {{.*}}[@child]
