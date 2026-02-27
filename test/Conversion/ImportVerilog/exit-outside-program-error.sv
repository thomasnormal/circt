// RUN: ! circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s
// REQUIRES: slang

module ExitOutsideProgramError;
  initial begin
    $exit;
  end
endmodule

// CHECK: error: $exit is only valid in program blocks
