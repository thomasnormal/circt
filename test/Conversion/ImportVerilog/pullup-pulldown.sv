// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test pullup and pulldown primitive support

// CHECK-LABEL: moore.module @test_pullup
module test_pullup;
  wire a;
  // CHECK: moore.constant 1 : l1
  // CHECK: %a = moore.assigned_variable
  pullup (a);
endmodule

// CHECK-LABEL: moore.module @test_pulldown
module test_pulldown;
  wire b;
  // CHECK: moore.constant 0 : l1
  // CHECK: %b = moore.assigned_variable
  pulldown (b);
endmodule

// CHECK-LABEL: moore.module @test_pullup_named
module test_pullup_named;
  wire c;
  // CHECK: moore.constant 1 : l1
  // CHECK: %c = moore.assigned_variable
  pullup p1 (c);
endmodule

// CHECK-LABEL: moore.module @test_multibit
module test_multibit;
  wire [7:0] d;
  // CHECK: moore.constant -1 : l8
  // CHECK: %d = moore.assigned_variable
  pullup (d);
endmodule
