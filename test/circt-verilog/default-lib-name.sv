// RUN: split-file %s %t
// RUN: not circt-verilog --parse-only --no-uvm-auto-include -l%t/lib-good.sv -lbar=%t/lib-bad.sv -Lfoo,bar %t/top.sv 2>&1 | FileCheck %s --check-prefix=NODEFAULT
// RUN: circt-verilog --parse-only --no-uvm-auto-include --defaultLibName=foo -l%t/lib-good.sv -lbar=%t/lib-bad.sv -Lfoo,bar %t/top.sv 2>&1 | FileCheck %s --check-prefix=WITHDEFAULT --allow-empty
// REQUIRES: slang

// NODEFAULT: error: unknown module 'unknown_mod'
// WITHDEFAULT-NOT: error:

//--- top.sv
module top;
  logic y;
  m u(.y(y));
endmodule

//--- lib-good.sv
module m(output logic y);
  assign y = 1'b1;
endmodule

//--- lib-bad.sv
module m(output logic y);
  unknown_mod u();
endmodule
