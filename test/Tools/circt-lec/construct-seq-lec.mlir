// RUN: circt-opt --construct-seq-lec="first-module=regA second-module=regB bound=3" %s | FileCheck %s

// Two equivalent registered adders: both add 1 to a register every clock cycle.
hw.module @regA(in %clk: !seq.clock, in %in: i8,
                in %r_state: i8, out out: i8, out r_next: i8)
    attributes {num_regs = 1 : i32, initial_values = [0 : i8]} {
  %one = hw.constant 1 : i8
  %sum = comb.add %r_state, %one : i8
  hw.output %sum, %sum : i8, i8
}

hw.module @regB(in %clk: !seq.clock, in %in: i8,
                in %r_state: i8, out out: i8, out r_next: i8)
    attributes {num_regs = 1 : i32, initial_values = [0 : i8]} {
  %one = hw.constant 1 : i8
  %sum = comb.add %one, %r_state : i8
  hw.output %sum, %sum : i8, i8
}

// CHECK: verif.bmc bound 6 num_regs 2 initial_values [0 : i8, 0 : i8]
// CHECK-SAME: init {
// CHECK:   seq.to_clock
// CHECK:   verif.yield
// CHECK: } loop {
// CHECK:   seq.from_clock
// CHECK:   comb.xor
// CHECK:   seq.to_clock
// CHECK:   verif.yield
// CHECK: } circuit {
// CHECK: ^bb0({{.*}}: !seq.clock, {{.*}}: i8, {{.*}}: i8, {{.*}}: i8):
// CHECK-DAG: comb.add
// CHECK-DAG: comb.add
// CHECK:   comb.icmp eq
// CHECK:   verif.assert
// CHECK:   verif.yield
// CHECK: }
