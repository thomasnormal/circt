// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test pullup and pulldown primitive support

// CHECK-LABEL: moore.module @test_pullup
module test_pullup;
  wire a;
  // CHECK: %a = moore.net wire : <l1>
  // CHECK: [[C1:%.+]] = moore.constant 1 : l1
  // CHECK: moore.assign %a, [[C1]] : l1
  pullup (a);
endmodule

// CHECK-LABEL: moore.module @test_pulldown
module test_pulldown;
  wire b;
  // CHECK: %b = moore.net wire : <l1>
  // CHECK: [[C0:%.+]] = moore.constant 0 : l1
  // CHECK: moore.assign %b, [[C0]] : l1
  pulldown (b);
endmodule

// CHECK-LABEL: moore.module @test_pullup_named
module test_pullup_named;
  wire c;
  // CHECK: %c = moore.net wire : <l1>
  // CHECK: [[C1:%.+]] = moore.constant 1 : l1
  // CHECK: moore.assign %c, [[C1]] : l1
  pullup p1 (c);
endmodule

// CHECK-LABEL: moore.module @test_multibit
module test_multibit;
  wire [7:0] d;
  // CHECK: %d = moore.net wire : <l8>
  // CHECK: [[CVAL:%.+]] = moore.constant 255 : l8
  // CHECK: moore.assign %d, [[CVAL]] : l8
  pullup (d);
endmodule
