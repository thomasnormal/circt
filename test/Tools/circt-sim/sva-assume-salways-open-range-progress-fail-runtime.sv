// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=50000000 2>&1 | FileCheck %s
// XFAIL: *
// CHECK: SVA assumption failed at time
// CHECK: SVA assumption failure(s)
// CHECK: exit code 1

// Runtime semantics: strong open-range always in a clocked assume must fail if
// simulation ends before the lower bound can be reached.
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
    @(posedge clk);
    @(posedge clk);
    $finish;
  end

  assume property (@(posedge clk) s_always [3:$] a);
endmodule
