// RUN: circt-verilog --lint-only --no-uvm-auto-include %s 2>&1 | FileCheck %s --check-prefix=NO-SUPPRESS
// RUN: circt-verilog --lint-only --no-uvm-auto-include --suppress-warnings=%s %s 2>&1 | FileCheck %s --check-prefix=SUPPRESS --allow-empty
// REQUIRES: slang

// NO-SUPPRESS: warning:
// NO-SUPPRESS: [-Windex-oob]
// SUPPRESS-NOT: warning:
// SUPPRESS-NOT: [-Windex-oob]

module top;
  logic [3:0] a;
  logic b;
  assign b = a[8];
endmodule
