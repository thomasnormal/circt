// RUN: circt-sim %s --top=test_func_recursion_args 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test that recursive func.call restores block argument values across nested
// calls. Without this, recursive calls overwrite the caller's argument values
// in the interpreter value map and produce incorrect results.

// CHECK: sum=10

func.func @sum_to(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %is_zero = arith.cmpi eq, %n, %c0 : i32
  cf.cond_br %is_zero, ^base, ^recurse

^base:
  return %c0 : i32

^recurse:
  %n_minus_1 = arith.subi %n, %c1 : i32
  %rec = func.call @sum_to(%n_minus_1) : (i32) -> i32
  %sum = arith.addi %n, %rec : i32
  return %sum : i32
}

hw.module @test_func_recursion_args() {
  %c1_i64 = hw.constant 1 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_pre = sim.fmt.literal "sum="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %t = llhd.int_to_time %c1_i64
    llhd.wait delay %t, ^bb1
  ^bb1:
    %c4 = arith.constant 4 : i32
    %result = func.call @sum_to(%c4) : (i32) -> i32
    %fmt_val = sim.fmt.dec %result : i32
    %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}

