// RUN: circt-verilog --ir-moore %s | FileCheck %s --check-prefix=MOORE
// RUN: circt-verilog --ir-hw %s | FileCheck %s --check-prefix=HW

// Test signal strength parsing for continuous assignments and lowering to LLHD

// MOORE-LABEL: moore.module @test_strength_basic
// HW-LABEL: hw.module @test_strength_basic
module test_strength_basic(input a, input b, output wire o);
  // MOORE: moore.assign %o, %{{.*}} strength(weak, weak) : l1
  // HW: llhd.drv %o, %a after %{{.*}} strength(weak, weak)
  assign (weak0, weak1) o = a;
  // MOORE: moore.assign %o, %{{.*}} strength(strong, strong) : l1
  // HW: llhd.drv %o, %b after %{{.*}} strength(strong, strong)
  assign (strong0, strong1) o = b;
endmodule

// MOORE-LABEL: moore.module @test_strength_pull
// HW-LABEL: hw.module @test_strength_pull
module test_strength_pull(input a, input b, output wire o);
  // MOORE: moore.assign %o, %{{.*}} strength(pull, pull) : l1
  // HW: llhd.drv %o, %a after %{{.*}} strength(pull, pull)
  assign (pull0, pull1) o = a;
  // MOORE: moore.assign %o, %{{.*}} strength(weak, weak) : l1
  // HW: llhd.drv %o, %b after %{{.*}} strength(weak, weak)
  assign (weak0, weak1) o = b;
endmodule

// MOORE-LABEL: moore.module @test_pullup_with_driver
// HW-LABEL: hw.module @test_pullup_with_driver
module test_pullup_with_driver(input clk, output wire o);
  wire w;
  // Pullup should have highz for strength0, and pull for strength1
  // MOORE: moore.assign %w, %{{.*}} strength(highz, pull) : l1
  // HW: llhd.drv %w, %{{.*}} after %{{.*}} strength(highz, pull)
  pullup (w);
  // MOORE: moore.assign %w, %{{.*}} strength(weak, weak) : l1
  // HW: llhd.drv %w, %clk after %{{.*}} strength(weak, weak)
  assign (weak0, weak1) w = clk;
  assign o = w;
endmodule

// MOORE-LABEL: moore.module @test_pulldown_with_driver
// HW-LABEL: hw.module @test_pulldown_with_driver
module test_pulldown_with_driver(input clk, output wire o);
  wire w;
  // Pulldown should have pull for strength0, and highz for strength1
  // MOORE: moore.assign %w, %{{.*}} strength(pull, highz) : l1
  // HW: llhd.drv %w, %{{.*}} after %{{.*}} strength(pull, highz)
  pulldown (w);
  // MOORE: moore.assign %w, %{{.*}} strength(weak, weak) : l1
  // HW: llhd.drv %w, %clk after %{{.*}} strength(weak, weak)
  assign (weak0, weak1) w = clk;
  assign o = w;
endmodule

// MOORE-LABEL: moore.module @test_pullup_strong
// HW-LABEL: hw.module @test_pullup_strong
module test_pullup_strong(input clk, output wire o);
  wire w;
  // MOORE: moore.assign %w, %{{.*}} strength(highz, strong) : l1
  // HW: llhd.drv %w, %{{.*}} after %{{.*}} strength(highz, strong)
  pullup (strong1) (w);
  // MOORE: moore.assign %w, %{{.*}} strength(weak, weak) : l1
  // HW: llhd.drv %w, %clk after %{{.*}} strength(weak, weak)
  assign (weak0, weak1) w = clk;
  assign o = w;
endmodule

// MOORE-LABEL: moore.module @test_supply_strength
// HW-LABEL: hw.module @test_supply_strength
module test_supply_strength(input a, input b, output wire o);
  // MOORE: moore.assign %o, %{{.*}} strength(supply, supply) : l1
  // HW: llhd.drv %o, %a after %{{.*}} strength(supply, supply)
  assign (supply0, supply1) o = a;
  // MOORE: moore.assign %o, %{{.*}} strength(weak, weak) : l1
  // HW: llhd.drv %o, %b after %{{.*}} strength(weak, weak)
  assign (weak0, weak1) o = b;
endmodule
