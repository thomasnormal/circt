// RUN: ! circt-translate --import-verilog %s 2>&1 | FileCheck %s
// REQUIRES: slang

module SvaInvalidClockingError(input logic a);
  // Invalid assertion clocking event expression must fail import; it must not
  // be silently dropped from the IR.
  // CHECK: error: expected a 1-bit integer
  assert property (@(1) a);
endmodule
