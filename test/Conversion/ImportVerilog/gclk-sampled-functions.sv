// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

// Test IEEE 1800-2017 Section 16.9.3 global clocking sampled value functions.
// The _gclk variants use the global clocking event and are semantically
// equivalent to their non-gclk counterparts for elaboration purposes.

module GclkSampledFunctions(input logic clk, a);
  // CHECK-LABEL: moore.module @GclkSampledFunctions
  global clocking @(posedge clk); endclocking

  // $rose_gclk - maps to $rose: bool_cast, past, not, and
  // CHECK: moore.bool_cast
  // CHECK-NEXT: moore.past {{.*}} delay 1
  // CHECK-NEXT: moore.not
  // CHECK: moore.and
  // CHECK: ltl.clock {{.*}} : i1
  // CHECK-NEXT: verif.assert
  assert property (@(posedge clk) $rose_gclk(a));

  // $fell_gclk - maps to $fell: bool_cast, not(current), past, and
  // CHECK: moore.bool_cast
  // CHECK-NEXT: moore.past {{.*}} delay 1
  // CHECK-NEXT: moore.not
  // CHECK: moore.and
  // CHECK: ltl.clock {{.*}} : i1
  // CHECK-NEXT: verif.assert
  assert property (@(posedge clk) $fell_gclk(a));

  // $stable_gclk - maps to $stable: past, eq
  // CHECK: moore.past {{.*}} delay 1
  // CHECK-NEXT: moore.eq
  // CHECK: ltl.clock {{.*}} : i1
  // CHECK-NEXT: verif.assert
  assert property (@(posedge clk) $stable_gclk(a));

  // $changed_gclk - maps to $changed: past, eq, not
  // CHECK: moore.past {{.*}} delay 1
  // CHECK-NEXT: moore.eq
  // CHECK-NEXT: moore.not
  // CHECK: ltl.clock {{.*}} : i1
  // CHECK-NEXT: verif.assert
  assert property (@(posedge clk) $changed_gclk(a));

  // $past_gclk - maps to $past: past
  // CHECK: moore.past {{.*}} delay 1
  // CHECK: ltl.clock {{.*}} : i1
  // CHECK-NEXT: verif.assert
  assert property (@(posedge clk) $past_gclk(a));

  // $future_gclk - maps to one-cycle future via ltl.delay.
  // CHECK: moore.to_builtin_bool
  // CHECK-NEXT: ltl.delay {{.*}}, 1, 0 : i1
  // CHECK: ltl.clock {{.*}} : !ltl.sequence
  // CHECK-NEXT: verif.assert
  assert property (@(posedge clk) $future_gclk(a));

  // $rising_gclk - maps to $rose
  // CHECK: moore.bool_cast
  // CHECK-NEXT: moore.past {{.*}} delay 1
  // CHECK-NEXT: moore.not
  // CHECK: moore.and
  // CHECK: ltl.clock {{.*}} : i1
  // CHECK-NEXT: verif.assert
  assert property (@(posedge clk) $rising_gclk(a));

  // $falling_gclk - maps to $fell
  // CHECK: moore.bool_cast
  // CHECK-NEXT: moore.past {{.*}} delay 1
  // CHECK-NEXT: moore.not
  // CHECK: moore.and
  // CHECK: ltl.clock {{.*}} : i1
  // CHECK-NEXT: verif.assert
  assert property (@(posedge clk) $falling_gclk(a));

  // $steady_gclk - maps to $stable
  // CHECK: moore.past {{.*}} delay 1
  // CHECK-NEXT: moore.eq
  // CHECK: ltl.clock {{.*}} : i1
  // CHECK-NEXT: verif.assert
  assert property (@(posedge clk) $steady_gclk(a));

  // $changing_gclk - maps to $changed
  // CHECK: moore.past {{.*}} delay 1
  // CHECK-NEXT: moore.eq
  // CHECK-NEXT: moore.not
  // CHECK: ltl.clock {{.*}} : i1
  // CHECK-NEXT: verif.assert
  assert property (@(posedge clk) $changing_gclk(a));

endmodule
