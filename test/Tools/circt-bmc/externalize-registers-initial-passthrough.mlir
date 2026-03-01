// RUN: circt-opt --externalize-registers %s | FileCheck %s

hw.module @top(in %clk: !seq.clock, in %in: i32, out out: i32) {
  %init0 = seq.initial () {
    %c5 = hw.constant 5 : i32
    seq.yield %c5 : i32
  } : () -> !seq.immutable<i32>

  // Forward the immutable value through a second initial op; this should still
  // resolve to the constant initializer.
  %init1 = seq.initial (%init0) {
  ^bb0(%arg0: i32):
    seq.yield %arg0 : i32
  } : (!seq.immutable<i32>) -> !seq.immutable<i32>

  %r = seq.compreg %in, %clk initial %init1 : i32
  hw.output %r : i32
}

// CHECK-LABEL: hw.module @top
// CHECK-SAME: in %r_state : i32
// CHECK-SAME: out r_next : i32
// CHECK-SAME: attributes {{{.*}}initial_values = [5 : i32], num_regs = 1 : i32}
