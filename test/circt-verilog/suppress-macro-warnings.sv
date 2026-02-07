// RUN: split-file %s %t
// RUN: circt-verilog --lint-only --no-uvm-auto-include %t/top.sv 2>&1 | FileCheck %s --check-prefix=NO-SUPPRESS
// RUN: circt-verilog --lint-only --no-uvm-auto-include --suppress-macro-warnings=%t/macro.svh %t/top.sv 2>&1 | FileCheck %s --check-prefix=SUPPRESS --allow-empty
// REQUIRES: slang

// NO-SUPPRESS: warning:
// NO-SUPPRESS: [-Windex-oob]
// SUPPRESS-NOT: warning:
// SUPPRESS-NOT: [-Windex-oob]

//--- top.sv
`include "macro.svh"
module top;
  logic [3:0] a;
  logic b;
  assign b = `OOB_IDX(a);
endmodule

//--- macro.svh
`define OOB_IDX(sig) sig[8]
