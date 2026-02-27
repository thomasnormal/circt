// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=90000000 2>&1 | FileCheck %s
// XFAIL: *
// CHECK-NOT: SVA assumption failed
// CHECK: Simulation completed

// Runtime semantics: strong open-range always in a clocked assume passes when
// lower-bound progress is achieved and the predicate remains true.
// FIXME: open-range `$` upper bounds in `s_always [n:$]` are currently
// rejected during parse in this flow.

module top;
  reg clk;
  reg a;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b1;
    repeat (6) @(posedge clk);
    $finish;
  end

  assume property (@(posedge clk) s_always [2:$] a);
endmodule
