// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Xcelium accepts width modifiers on these specifiers and ignores width with a
// warning. Keep import compatibility by accepting them as well.

// CHECK-LABEL: moore.module @format_width_ignored_compat
module format_width_ignored_compat;
  logic [3:0] s;
  int i;
  class C;
  endclass
  C h;

  initial begin
    $display("c=%4c", i);
    $display("p=%-4p", h);
    $display("u=%0u", s);
    $display("z=%4z", s);
    $display("v=%-4v", s);
    $display("m=%0m");
    $display("l=%4l");
  end

  // CHECK: moore.fmt.char
  // CHECK: moore.fmt.class
  // CHECK-COUNT-3: moore.fmt.int {{.*}}binary
  // CHECK: moore.fmt.literal "m="
  // CHECK: moore.builtin.display
  // CHECK: moore.fmt.literal "l="
  // CHECK: moore.builtin.display
endmodule
