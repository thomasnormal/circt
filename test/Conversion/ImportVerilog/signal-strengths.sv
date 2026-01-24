// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test signal strength parsing for continuous assignments

// CHECK-LABEL: moore.module @test_strength_basic
module test_strength_basic(input a, input b, output wire o);
  // CHECK: moore.assign %o, %{{.*}} strength(weak, weak) : l1
  assign (weak0, weak1) o = a;
  // CHECK: moore.assign %o, %{{.*}} strength(strong, strong) : l1
  assign (strong0, strong1) o = b;
endmodule

// CHECK-LABEL: moore.module @test_strength_pull
module test_strength_pull(input a, input b, output wire o);
  // CHECK: moore.assign %o, %{{.*}} strength(pull, pull) : l1
  assign (pull0, pull1) o = a;
  // CHECK: moore.assign %o, %{{.*}} strength(weak, weak) : l1
  assign (weak0, weak1) o = b;
endmodule

// CHECK-LABEL: moore.module @test_pullup_with_driver
module test_pullup_with_driver(input clk, output wire o);
  wire w;
  // Pullup should have highz for strength0, and pull for strength1
  // CHECK: moore.assign %w, %{{.*}} strength(highz, pull) : l1
  pullup (w);
  // CHECK: moore.assign %w, %{{.*}} strength(weak, weak) : l1
  assign (weak0, weak1) w = clk;
  assign o = w;
endmodule

// CHECK-LABEL: moore.module @test_pulldown_with_driver
module test_pulldown_with_driver(input clk, output wire o);
  wire w;
  // Pulldown should have pull for strength0, and highz for strength1
  // CHECK: moore.assign %w, %{{.*}} strength(pull, highz) : l1
  pulldown (w);
  // CHECK: moore.assign %w, %{{.*}} strength(weak, weak) : l1
  assign (weak0, weak1) w = clk;
  assign o = w;
endmodule

// CHECK-LABEL: moore.module @test_pullup_strong
module test_pullup_strong(input clk, output wire o);
  wire w;
  // CHECK: moore.assign %w, %{{.*}} strength(highz, strong) : l1
  pullup (strong1) (w);
  // CHECK: moore.assign %w, %{{.*}} strength(weak, weak) : l1
  assign (weak0, weak1) w = clk;
  assign o = w;
endmodule

// CHECK-LABEL: moore.module @test_supply_strength
module test_supply_strength(input a, input b, output wire o);
  // CHECK: moore.assign %o, %{{.*}} strength(supply, supply) : l1
  assign (supply0, supply1) o = a;
  // CHECK: moore.assign %o, %{{.*}} strength(weak, weak) : l1
  assign (weak0, weak1) o = b;
endmodule
