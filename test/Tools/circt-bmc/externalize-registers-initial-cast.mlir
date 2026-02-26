// RUN: circt-opt --externalize-registers %s | FileCheck %s

hw.module @top(in %clk: !seq.clock, in %in: i32, out out: i32) {
  %init = seq.initial () {
    %c5 = hw.constant 5 : i32
    seq.yield %c5 : i32
  } : () -> !seq.immutable<i32>
  %init_cast = builtin.unrealized_conversion_cast %init : !seq.immutable<i32> to !seq.immutable<i32>
  %r = seq.compreg %in, %clk initial %init_cast : i32
  hw.output %r : i32
}

// CHECK-LABEL: hw.module @top
// CHECK-SAME: in %r_state : i32
// CHECK-SAME: out r_next : i32
// CHECK-SAME: attributes {{{.*}}initial_values = [5 : i32], num_regs = 1 : i32}
