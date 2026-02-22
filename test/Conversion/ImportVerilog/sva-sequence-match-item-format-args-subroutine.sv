// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-translate --import-verilog %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemFormatArgsSubroutine(input logic clk, a);
  sequence s;
    int fd;
    (1, fd = 0, $display("a=%0d", a), $fdisplay(fd, "a=%0d", a)) ##1 a;
  endsequence

  // Match-item display/fdisplay should preserve format arguments, not just
  // emit task-name marker literals.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemFormatArgsSubroutine
  // CHECK: moore.read %a
  // CHECK: moore.fmt.int decimal
  // CHECK: moore.builtin.display
  // CHECK: moore.builtin.fwrite
  // CHECK: verif.assert
  assert property (@(posedge clk) s);
endmodule

// DIAG-NOT: ignoring system subroutine `$display`
// DIAG-NOT: ignoring system subroutine `$fdisplay`
