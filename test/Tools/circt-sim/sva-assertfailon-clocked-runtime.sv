// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=20000000 > %t.out 2>&1
// RUN: /usr/bin/grep -q "clocked_failon_done" %t.out
// RUN: /usr/bin/grep -q "SVA assertion failed:" %t.out
//
// $assertfailon should re-enable clocked assertion failure diagnostics after
// an earlier $assertfailoff.

module top;
  reg clk = 0;
  reg a = 0;

  always #1 clk = ~clk;

  initial begin
    $assertfailoff;
    #2;
    $assertfailon;
    #2;
    $display("clocked_failon_done");
    $finish;
  end

  ap: assert property (@(posedge clk) a);
endmodule
