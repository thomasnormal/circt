// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=20000000 > %t.out 2>&1
// RUN: /usr/bin/grep -q "after_assertoff_on_transition" %t.out
// RUN: /usr/bin/grep -q "Simulation completed" %t.out
// RUN: not /usr/bin/grep -q "\\[SVA\\] assertion failed" %t.out

module top;
  reg clk;
  reg a;

  initial begin
    clk = 1'b0;
    a = 1'b0;
    $assertoff;
    #4;
    a = 1'b1;
    $asserton;
    $display("after_assertoff_on_transition");
    #4;
    $finish;
  end

  always #1 clk = ~clk;

  ap: assert property (@(posedge clk) a);
endmodule
