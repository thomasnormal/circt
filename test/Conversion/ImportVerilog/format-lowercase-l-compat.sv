// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Compatibility test: allow `%l` in display format strings.
// IEEE defines %l as library binding info; ImportVerilog currently lowers it
// using the same scope-path fallback behavior as `%m`.

module FormatLowercaseLCompat;
  // CHECK-LABEL: moore.module @FormatLowercaseLCompat
  // CHECK: moore.fmt.literal "FormatLowercaseLCompat"
  // CHECK: moore.fmt.literal "\0A"
  // CHECK: moore.fmt.concat
  // CHECK: moore.builtin.display
  initial $display("%l");

  // CHECK: moore.fmt.literal "path="
  // CHECK: moore.fmt.literal "FormatLowercaseLCompat"
  // CHECK: moore.fmt.literal "\0A"
  // CHECK: moore.fmt.concat
  // CHECK: moore.builtin.display
  initial $display("path=%l");
endmodule
