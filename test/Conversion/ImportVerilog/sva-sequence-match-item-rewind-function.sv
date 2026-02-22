// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-translate --import-verilog %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemRewindFunction(input logic clk, a);
  sequence s;
    int rc;
    (1, rc = $rewind(1)) ##1 a;
  endsequence

  // `$rewind` used as a value-returning call in match-item assignments should
  // preserve side effects and produce a usable return value.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemRewindFunction
  // CHECK: moore.builtin.rewind
  // CHECK: verif.assert
  assert property (@(posedge clk) s);
endmodule

// DIAG-NOT: unsupported system call `$rewind`
