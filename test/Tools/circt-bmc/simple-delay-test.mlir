// RUN: circt-opt %s --externalize-registers --lower-to-bmc="top-module=simple_counter bound=10" | FileCheck %s

// Simple counter test: verify that a register increments correctly
// This tests basic BMC functionality with registers (1-step state tracking)

// CHECK-LABEL: func.func @simple_counter
hw.module @simple_counter(
  in %clk: !seq.clock,
  in %en: i1,
  out count: i8
) {
  %c1_i8 = hw.constant 1 : i8
  %c0_i8 = hw.constant 0 : i8

  // Counter register
  %counter = seq.compreg %next_count, %clk : i8

  // Increment logic
  %incremented = comb.add %counter, %c1_i8 : i8

  // Next value: increment if enabled, else hold
  %next_count = comb.mux %en, %incremented, %counter : i8

  // Property: counter should never overflow (stay < 100)
  %c100_i8 = hw.constant 100 : i8
  %not_overflow = comb.icmp ult %counter, %c100_i8 : i8
  verif.assert %not_overflow : i1

  hw.output %counter : i8
}

// This property can be violated if enable is high for 100+ cycles
// BMC should be able to find a counterexample or prove it within bound

// CHECK: verif.bmc bound 20 num_regs 1
// CHECK: init {
// CHECK:   seq.to_clock
// CHECK: }
// CHECK: loop {
// CHECK:   seq.from_clock
// CHECK:   comb.xor
// CHECK:   seq.to_clock
// CHECK: }
// CHECK: circuit {
// CHECK:   comb.add
// CHECK:   comb.mux
// CHECK:   comb.icmp
// CHECK:   verif.assert
// CHECK: }
