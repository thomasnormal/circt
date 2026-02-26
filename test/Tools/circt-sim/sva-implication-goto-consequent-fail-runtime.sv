// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=120000000 2>&1 | FileCheck %s
// CHECK-NOT: SVA assertion failed at time 15000000 fs
// CHECK: SVA assertion failed at time 65000000 fs
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Gap test: implication with a variable-length consequent should not fail at
// the trigger cycle. This sequence remains open-ended, so an unsatisfied
// obligation is conservatively reported at end-of-run.

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

    @(posedge clk); // 5ns
    a = 1;

    @(posedge clk); // 15ns: antecedent samples true
    a = 0;
    b = 1;

    @(posedge clk); // 25ns: first b hit
    b = 0;

    @(posedge clk); // 35ns
    b = 1;

    @(posedge clk); // 45ns: second b hit
    b = 0;

    // c remains low, leaving the implication obligation unresolved.
    @(posedge clk); // 55ns
    @(posedge clk); // 65ns
    $finish;
  end

  assert property (@(posedge clk) a |-> (b[->2] ##1 c));
endmodule
