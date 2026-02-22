// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-translate --import-verilog %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemStacktraceFunction(input logic clk, a);
  sequence s;
    string st;
    (1, st = $stacktrace) ##1 a;
  endsequence

  // Value-returning $stacktrace should lower in match-item assignment RHS.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemStacktraceFunction
  // CHECK: moore.constant_string
  // CHECK: verif.assert
  assert property (@(posedge clk) s);
endmodule

// DIAG-NOT: unsupported system call `$stacktrace`
