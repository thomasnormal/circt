// RUN: split-file %s %t
// RUN: circt-verilog --parse-only --no-uvm-auto-include -lL1=%t/lib-good.sv -lL2=%t/lib-bad.sv -LL1,L2 %t/top.sv 2>&1 | FileCheck %s --check-prefix=GOOD --allow-empty
// RUN: not circt-verilog --parse-only --no-uvm-auto-include -lL1=%t/lib-good.sv -lL2=%t/lib-bad.sv -LL2,L1 %t/top.sv 2>&1 | FileCheck %s --check-prefix=BAD
// REQUIRES: slang

// GOOD-NOT: error:

// BAD: error: unknown module 'unknown_mod'

//--- top.sv
module top;
  logic y;
  m u(.y(y));
endmodule

//--- lib-good.sv
module m(output logic y);
  assign y = 1'b0;
endmodule

//--- lib-bad.sv
module m(output logic y);
  unknown_mod u();
endmodule
