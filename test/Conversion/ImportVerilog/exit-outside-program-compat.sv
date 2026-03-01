// RUN: circt-verilog --no-uvm-auto-include --allow-exit-outside-program --ir-moore %s 2>&1 | FileCheck %s
// REQUIRES: slang

module ExitOutsideProgramCompat;
  initial begin
    $exit;
  end
endmodule

// CHECK: warning: $exit outside program block lowered as $finish due to --allow-exit-outside-program
// CHECK: moore.builtin.finish
// CHECK: moore.unreachable
