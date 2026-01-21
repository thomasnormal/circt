// RUN: not circt-verilog --no-uvm-auto-include %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-UVM
// REQUIRES: slang

// CHECK-NO-UVM: error:
// CHECK-NO-UVM-SAME: uvm_pkg

module top;
  import uvm_pkg::*;
endmodule
