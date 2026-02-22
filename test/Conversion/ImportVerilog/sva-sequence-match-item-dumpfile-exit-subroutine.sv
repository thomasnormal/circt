// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemDumpfileExitSubroutine(input logic clk, a);
  sequence s_dump;
    (1, $dumpfile("trace.vcd")) ##1 a;
  endsequence
  sequence s_exit;
    (1, $exit) ##1 a;
  endsequence

  // Match-item dump/control tasks should preserve side effects.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemDumpfileExitSubroutine
  // CHECK: moore.builtin.display
  // CHECK: {circt.dumpfile = "trace.vcd"}
  // CHECK: verif.assert
  assert property (@(posedge clk) s_dump);

  // CHECK: moore.builtin.finish 0
  // CHECK: verif.assert
  assert property (@(posedge clk) s_exit);
endmodule
