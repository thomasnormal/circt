// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS_IMPLICATION_GOTO

// Ensure implication consequents containing goto-repeat can satisfy later
// without an immediate fail at antecedent time.

module top;
  reg clk;
  reg trig;
  reg b;
  reg c;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    trig = 0;
    b = 0;
    c = 0;

    @(negedge clk);
    trig = 1;

    // first b hit for b[->1]
    @(negedge clk);
    trig = 0;
    b = 1;

    // c one cycle after the hit
    @(negedge clk);
    b = 0;
    c = 1;

    @(posedge clk);
    $display("SVA_PASS_IMPLICATION_GOTO");
    $finish;
  end

  assert property (@(posedge clk) trig |-> ((b[->1]) ##1 c));
endmodule
