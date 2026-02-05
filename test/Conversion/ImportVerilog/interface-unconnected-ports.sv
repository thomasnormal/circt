// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

interface SimpleIf(input logic in_sig, output logic out_sig);
  logic internal;
endinterface

module top;
  SimpleIf ifc();
endmodule

// CHECK: moore.interface @SimpleIf
// CHECK: moore.interface.instance @SimpleIf
