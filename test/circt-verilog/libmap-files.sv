// RUN: split-file %s %t
// RUN: circt-verilog -E --no-uvm-auto-include %t/top.sv | FileCheck %s --check-prefix=NO-LIBMAP
// RUN: circt-verilog -E --no-uvm-auto-include --libmap %t/lib.map %t/top.sv | FileCheck %s --check-prefix=WITH-LIBMAP
// REQUIRES: slang

// NO-LIBMAP: module top;
// NO-LIBMAP-NOT: module map_lib_mod;

// WITH-LIBMAP: module top;
// WITH-LIBMAP: module map_lib_mod;

//--- top.sv
module top;
  map_lib_mod u();
endmodule

//--- lib.map
library maplib lib.sv;

//--- lib.sv
module map_lib_mod;
endmodule
