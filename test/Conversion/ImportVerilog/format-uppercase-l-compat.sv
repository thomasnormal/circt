// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Compatibility test: allow `%L` in display format strings as an alias of `%m`.
module FormatUppercaseLCompat;
  // CHECK-LABEL: moore.module @FormatUppercaseLCompat
  // CHECK: moore.fmt.literal "FormatUppercaseLCompat"
  // CHECK: moore.fmt.literal "\0A"
  // CHECK: moore.fmt.concat
  // CHECK: moore.builtin.display
  initial $display("%L");

  // CHECK: moore.fmt.literal "path="
  // CHECK: moore.fmt.literal "FormatUppercaseLCompat"
  // CHECK: moore.fmt.literal "\0A"
  // CHECK: moore.fmt.concat
  // CHECK: moore.builtin.display
  initial $display("path=%L");
endmodule
