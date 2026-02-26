// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=20000000 > %t.out 2>&1
// RUN: /usr/bin/grep -q "clocked_failoff_done" %t.out
// RUN: /usr/bin/grep -q "SVA assertion failure(s)" %t.out
// RUN: not /usr/bin/grep -q "SVA assertion failed:" %t.out
//
// $assertfailoff should suppress clocked assertion failure messages while
// preserving assertion failure accounting / non-zero exit status.

module top;
  reg clk = 0;
  reg a = 0;

  always #1 clk = ~clk;

  initial begin
    $assertfailoff;
    #4;
    $display("clocked_failoff_done");
    $finish;
  end

  ap: assert property (@(posedge clk) a);
endmodule
