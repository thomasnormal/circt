// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemFileControlSubroutine(input logic clk, a);
  int fd;
  sequence s;
    int x;
    (1, x = 0, $fflush(fd), $fclose(fd)) ##1 (1, x += 1) ##1 (x == 1);
  endsequence

  // File-control match-item subroutines should preserve side effects.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemFileControlSubroutine
  // CHECK: moore.builtin.fflush
  // CHECK: moore.builtin.fclose
  // CHECK: verif.assert
  assert property (@(posedge clk) a |-> s);
endmodule
