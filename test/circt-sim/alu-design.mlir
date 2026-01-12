// RUN: circt-sim %s --top=ALU --mode=analyze 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// A simple ALU design for testing combinational logic simulation.

// CHECK: === Design Analysis ===
// CHECK: Modules:
// CHECK: Ports:

// ALU operation codes
// 00 = ADD
// 01 = SUB
// 10 = AND
// 11 = OR

hw.module @ALU(in %a: i32, in %b: i32, in %op: i2, out result: i32, out zero: i1) {
  %c0_i32 = hw.constant 0 : i32
  %c1_i2 = hw.constant 1 : i2
  %c2_i2 = hw.constant 2 : i2
  %c3_i2 = hw.constant 3 : i2

  // Compute all possible results
  %add_result = comb.add %a, %b : i32
  %sub_result = comb.sub %a, %b : i32
  %and_result = comb.and %a, %b : i32
  %or_result = comb.or %a, %b : i32

  // Operation select comparisons
  %is_add = comb.icmp eq %op, %c0_i32 : i2
  %is_sub = comb.icmp eq %op, %c1_i2 : i2
  %is_and = comb.icmp eq %op, %c2_i2 : i2
  %is_or = comb.icmp eq %op, %c3_i2 : i2

  // Mux to select the result based on op
  %mux_1 = comb.mux %is_add, %add_result, %sub_result : i32
  %mux_2 = comb.mux %is_and, %and_result, %mux_1 : i32
  %mux_3 = comb.mux %is_or, %or_result, %mux_2 : i32

  // Zero flag
  %zero_flag = comb.icmp eq %mux_3, %c0_i32 : i32

  hw.output %mux_3, %zero_flag : i32, i1
}

// Top-level testbench
hw.module @ALU_TB(in %clk: !seq.clock, in %rst: i1) {
  %c10_i32 = hw.constant 10 : i32
  %c5_i32 = hw.constant 5 : i32
  %c0_i2 = hw.constant 0 : i2

  %result, %zero = hw.instance "dut" @ALU(a: %c10_i32: i32, b: %c5_i32: i32, op: %c0_i2: i2) -> (result: i32, zero: i1)
}
