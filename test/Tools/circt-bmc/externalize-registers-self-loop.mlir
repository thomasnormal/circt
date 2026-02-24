// RUN: circt-opt --externalize-registers %s | FileCheck %s

// CHECK-LABEL: hw.module @self_loop_compreg(
// CHECK-SAME: in [[STATE:%[^:]+]] : i1
// CHECK: hw.output [[STATE]], [[STATE]] : i1, i1
hw.module @self_loop_compreg(in %clk: !seq.clock, out out: i1) {
  %reg = seq.compreg %reg, %clk : i1
  hw.output %reg : i1
}

// CHECK-LABEL: hw.module @self_loop_compreg_reset(
// CHECK-SAME: in [[RST:%[^:]+]] : i1
// CHECK-SAME: in [[STATE:%[^:]+]] : i1
// CHECK: [[NEXT:%[^ ]+]] = comb.mux [[RST]], [[STATE]], [[STATE]] : i1
// CHECK: hw.output [[STATE]], [[NEXT]] : i1, i1
hw.module @self_loop_compreg_reset(in %clk: !seq.clock, in %rst: i1, out out: i1) {
  %reg = seq.compreg %reg, %clk reset %rst, %reg : i1
  hw.output %reg : i1
}

// CHECK-LABEL: hw.module @self_loop_firreg(
// CHECK-SAME: in [[STATE:%[^:]+]] : i1
// CHECK: hw.output [[STATE]], [[STATE]] : i1, i1
hw.module @self_loop_firreg(in %clk: !seq.clock, out out: i1) {
  %reg = seq.firreg %reg clock %clk : i1
  hw.output %reg : i1
}
