// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemFileSubroutine(input logic clk, a);
  int fd;
  sequence s;
    int x;
    (1, x = 0, $fdisplay(fd, "d"), $fwrite(fd, "w"), $fstrobe(fd, "s"),
     $fmonitor(fd, "m")) ##1 (1, x += 1) ##1 (x == 1);
  endsequence

  // File-oriented match-item subroutines should preserve side effects.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemFileSubroutine
  // CHECK: moore.builtin.fwrite
  // CHECK: moore.builtin.fwrite
  // CHECK: moore.builtin.fstrobe
  // CHECK: moore.builtin.fmonitor
  // CHECK: verif.assert
  assert property (@(posedge clk) a |-> s);
endmodule
