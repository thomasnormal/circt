// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemStacktraceFunction(input logic clk, a);
  sequence s;
    string st;
    (1, st = $stacktrace) ##1 a;
  endsequence

  // Value-returning $stacktrace in a dead local assignment should not prevent
  // sequence lowering.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemStacktraceFunction
  // CHECK: moore.bool_cast
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) s);
endmodule

// DIAG-NOT: unsupported system call `$stacktrace`
