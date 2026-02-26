// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS: always [1:$] property remained satisfied
// CHECK-NOT: SVA assertion failed
// CHECK-NOT: SVA assertion failure(s)

// Regression: weak eventually inside lowered always [m:$] must not evaluate to
// immediate true at each sample. This design keeps the inner property vacuously
// true, so the assertion should pass.

module top;
  reg clk;
  reg a;
  reg b;

  property p;
    @(posedge clk) a |-> b;
  endproperty

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 0;
    b = 0;
    repeat (5) @(posedge clk);
    $display("SVA_PASS: always [1:$] property remained satisfied");
    $finish;
  end

  assert property (@(posedge clk) always [1:$] p);
endmodule
