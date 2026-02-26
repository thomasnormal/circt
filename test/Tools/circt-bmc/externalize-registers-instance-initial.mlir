// RUN: circt-opt --externalize-registers %s | FileCheck %s

hw.module @init_emitter(out out: !seq.immutable<i32>) {
  %init = seq.initial () {
    %c7 = hw.constant 7 : i32
    seq.yield %c7 : i32
  } : () -> !seq.immutable<i32>
  hw.output %init : !seq.immutable<i32>
}

hw.module @top(in %clk: !seq.clock, in %in: i32, out out: i32) {
  %init = hw.instance "init" @init_emitter () -> (out: !seq.immutable<i32>)
  %r = seq.compreg %in, %clk initial %init : i32
  hw.output %r : i32
}

// CHECK-LABEL: hw.module @top
// CHECK-SAME: in %r_state : i32
// CHECK-SAME: out r_next : i32
// CHECK-SAME: attributes {{{.*}}initial_values = [7 : i32], num_regs = 1 : i32}
