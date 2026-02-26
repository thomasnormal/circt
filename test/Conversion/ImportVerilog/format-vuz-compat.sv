// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Compatibility test for additional standard display format specifiers
// that should not fail import: %u, %z, %v.

// CHECK-LABEL: moore.module @FormatUCompat
// CHECK: moore.fmt.int binary
// CHECK: moore.builtin.display
module FormatUCompat;
  logic [3:0] s;
  initial $display("u=%u", s);
endmodule

// CHECK-LABEL: moore.module @FormatZCompat
// CHECK: moore.fmt.int binary
// CHECK: moore.builtin.display
module FormatZCompat;
  logic [3:0] s;
  initial $display("z=%z", s);
endmodule

// CHECK-LABEL: moore.module @FormatVCompat
// CHECK: moore.fmt.int binary
// CHECK: moore.builtin.display
module FormatVCompat;
  logic [3:0] s;
  initial $display("v=%v", s);
endmodule
