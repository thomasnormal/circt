// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s --check-prefix=CHECK-MOORE
// REQUIRES: slang

module GclkGlobalClocking(input logic clk, a);
  // CHECK-LABEL: moore.module @GclkGlobalClocking
  global clocking @(posedge clk); endclocking

  // In unclocked properties, *_gclk sampled-value calls must still use the
  // global clocking event.
  // CHECK-COUNT-3: ltl.clock
  assert property ($rose_gclk(a));
  assert property ($past_gclk(a));
  assert property ($future_gclk(a));

  // CHECK-MOORE-LABEL: moore.module @GclkGlobalClocking
  // CHECK-MOORE-COUNT-3: verif.clocked_assert
  // CHECK-MOORE-NOT: verif.assert
endmodule
