// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemMonitorStrobeSubroutine(input logic clk, a);
  sequence s;
    int x;
    (1, x = 0, $strobe("s"), $monitor("m"), $monitoron(), $monitoroff(),
     $write("w")) ##1 (1, x += 1) ##1 (x == 1);
  endsequence

  // Match-item monitor/strobe family calls should preserve side effects.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemMonitorStrobeSubroutine
  // CHECK: moore.builtin.display
  // CHECK: moore.builtin.monitor
  // CHECK: moore.builtin.monitoron
  // CHECK: moore.builtin.monitoroff
  // CHECK: verif.assert
  assert property (@(posedge clk) a |-> s);
endmodule
