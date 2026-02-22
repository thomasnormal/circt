// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemTimeformatDumpcontrolSubroutine(input logic clk, a);
  sequence s;
    (1, $timeformat(-9, 2, " ns", 20), $dumpoff, $dumpon, $dumpflush,
     $dumpall, $printtimescale) ##1 a;
  endsequence

  // Match-item timeformat and dump-control tasks should no longer be ignored.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemTimeformatDumpcontrolSubroutine
  // CHECK: moore.builtin.timeformat
  // CHECK: moore.builtin.printtimescale
  // CHECK: verif.assert
  assert property (@(posedge clk) s);
endmodule
