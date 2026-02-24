// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// Test the full AOT compile-then-run pipeline for a simple func.func.
//
// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
//
// SIM: out=200
//
// COMPILED: Loaded 1 compiled functions: 1 native-eligible, 0 intercepted, 0 skipped (has call_indirect)
// COMPILED: out=200

// A pure arithmetic function — compilable by circt-sim-compile.
func.func @mul_i32(%a: i32, %b: i32) -> i32 {
  %c = arith.muli %a, %b : i32
  return %c : i32
}

hw.module @test() {
  %c10_i32 = hw.constant 10 : i32
  %c20_i32 = hw.constant 20 : i32
  %c5_i64  = hw.constant 5000000 : i64
  %c15_i64 = hw.constant 15000000 : i64
  %fmt_prefix = sim.fmt.literal "out="
  %fmt_nl     = sim.fmt.literal "\0A"

  // At t=5ns: call mul_i32(10, 20) = 200, print it, then halt.
  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^print
  ^print:
    %r = func.call @mul_i32(%c10_i32, %c20_i32) : (i32, i32) -> i32
    %fmt_val = sim.fmt.dec %r : i32
    %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    llhd.halt
  }

  // Terminator at t=15ns.
  llhd.process {
    %d = llhd.int_to_time %c15_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
