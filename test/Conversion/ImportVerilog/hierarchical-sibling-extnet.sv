// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module Top(input i, output o);
  // CHECK-LABEL: moore.module @Top
  // CHECK: moore.instance "A" @A(B.x: {{%.+}}: !moore.ref<l1>)
  A A();
  B B();
  assign A.i = i;
  assign o = B.o;
endmodule

// CHECK-LABEL: moore.module private @A(
// CHECK-SAME: in %B.x : !moore.ref<l1>
module A;
  wire i;
  assign B.x = i;
endmodule

// CHECK-LABEL: moore.module private @B(
// CHECK-SAME: out x : !moore.ref<l1>
module B;
  wire x;
  wire o;
  assign o = x;
endmodule
