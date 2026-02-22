// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-translate --import-verilog %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemFerrorLocalOutput(input logic clk, a);
  sequence s;
    int rc;
    string ferr;
    (1, rc = $ferror(0, ferr)) ##1 a;
  endsequence

  // Local assertion vars should be usable as output-arg lvalues in
  // value-returning match-item function calls.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemFerrorLocalOutput
  // CHECK: moore.builtin.ferror
  // CHECK: verif.assert
  assert property (@(posedge clk) s);
endmodule

// DIAG-NOT: no lvalue generated for LocalAssertionVar
// DIAG-NOT: local assertion variable referenced before assignment
