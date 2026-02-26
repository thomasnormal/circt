// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=60000000 >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// CHECK-NOT: SVA assertion failed at time 15000000 fs
// CHECK: SVA assertion failed at time 25000000 fs
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: for `a |-> (b ##1 c)`, when `a` is sampled high at a
// given edge and `b` is true at that same edge, `c` must be true on the next
// sampled edge. A failure should be reported at that next edge.

module top;
  reg clk;
  reg a;
  reg b;
  reg c;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 0;
    b = 0;
    c = 0;
    @(posedge clk);
    a = 1;
    b = 1;
    c = 0;
    @(posedge clk);
    a = 0;
    b = 0;
    c = 0;
    @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) a |-> (b ##1 c));
endmodule
