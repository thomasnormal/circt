// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Xcelium accepts %n / %N in display formatting with an argument.
// Keep import compatibility by lowering them through binary integer formatting.

// CHECK-LABEL: moore.module @format_n_compat
module format_n_compat;
  int x;

  initial begin
    $display("n=%n", x);
    $display("N=%N", x);
  end

  // CHECK: moore.fmt.int binary
  // CHECK: moore.builtin.display
  // CHECK: moore.fmt.int binary
  // CHECK: moore.builtin.display
endmodule
