// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=20000000 > %t.out 2>&1
// RUN: /usr/bin/grep -q "immediate_failoff_done" %t.out
// RUN: not /usr/bin/grep -q "\\[circt-sim\\] Assertion failed" %t.out
//
// $assertfailoff should suppress diagnostics for immediate assertions even when
// no explicit fail action block is present.

module top;
  reg clk = 0;
  reg a = 0;

  always #1 clk = ~clk;

  always @(posedge clk)
    assert (a == 1);

  initial begin
    $assertfailoff;
    #4;
    $display("immediate_failoff_done");
    $finish;
  end
endmodule
