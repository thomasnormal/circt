// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-translate --import-verilog %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemWriteMemSubroutine(input logic clk, a);
  logic [7:0] mem [0:3];

  sequence s;
    (1, $writememb("mem.bin", mem), $writememh("mem.hex", mem)) ##1 a;
  endsequence

  // Match-item memory dump tasks should preserve side effects.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemWriteMemSubroutine
  // CHECK: moore.builtin.writememb
  // CHECK: moore.builtin.writememh
  // CHECK: verif.assert
  assert property (@(posedge clk) s);
endmodule

// DIAG-NOT: ignoring system subroutine `$writememb`
// DIAG-NOT: ignoring system subroutine `$writememh`
