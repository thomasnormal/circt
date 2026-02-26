// RUN: circt-sim-compile %s -o %t.default.so 2>&1 | FileCheck %s --check-prefix=COMPILE-DEFAULT
// RUN: circt-sim %s --compiled=%t.default.so 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: env CIRCT_AOT_ALLOW_NATIVE_MOORE_DELAY_CALLERS=1 circt-sim-compile %s -o %t.optin.so 2>&1 | FileCheck %s --check-prefix=COMPILE-OPTIN

// Regression: native functions that directly call __moore_delay must be
// demoted by default. __moore_delay is handled via interpreter interception.
//
// COMPILE-DEFAULT: [circt-sim-compile] Functions: 3 total, 1 external, 0 rejected, 2 compilable
// COMPILE-DEFAULT: [circt-sim-compile] Demoted 1 intercepted functions to trampolines
// COMPILE-DEFAULT: [circt-sim-compile] 1 functions + 0 processes ready for codegen
//
// DEFAULT: Loaded 1 compiled functions: 1 native-dispatched, 0 not-native-dispatched, 0 intercepted
// DEFAULT: out=7
//
// COMPILE-OPTIN: [circt-sim-compile] Functions: 3 total, 1 external, 0 rejected, 2 compilable
// COMPILE-OPTIN: [circt-sim-compile] 2 functions + 0 processes ready for codegen

llvm.func @__moore_delay(i64)

func.func @delay_wrapper() -> i32 {
  %c0_i64 = arith.constant 0 : i64
  %c7_i32 = arith.constant 7 : i32
  llvm.call @__moore_delay(%c0_i64) : (i64) -> ()
  return %c7_i32 : i32
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}

hw.module @test() {
  %c5_i64 = hw.constant 5000000 : i64
  %c15_i64 = hw.constant 15000000 : i64
  %fmt_prefix = sim.fmt.literal "out="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^print
  ^print:
    %r = func.call @delay_wrapper() : () -> i32
    %fmt_val = sim.fmt.dec %r : i32
    %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %c15_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
