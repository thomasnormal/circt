// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=60000000 >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// CHECK-NOT: SVA assertion failed at time 5000000 fs
// CHECK: SVA assertion failed at time 15000000 fs
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: `##1 a` should not fail on the first sampled edge; the
// first decidable obligation appears one cycle later.

module top;
  reg clk;
  reg a;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b0;
    @(posedge clk);
    @(posedge clk);
    @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) ##1 a);
endmodule
