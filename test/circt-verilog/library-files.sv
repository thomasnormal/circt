// RUN: split-file %s %t
// RUN: circt-verilog -E --no-uvm-auto-include %t/top.sv | FileCheck %s --check-prefix=NO-LIB
// RUN: circt-verilog -E --no-uvm-auto-include -l%t/lib.sv %t/top.sv | FileCheck %s --check-prefix=WITH-LIB
// REQUIRES: slang

// NO-LIB: module top;
// NO-LIB-NOT: module library_mod;

// WITH-LIB: module top;
// WITH-LIB: module library_mod;

//--- top.sv
module top;
  library_mod u();
endmodule

//--- lib.sv
module library_mod;
endmodule
