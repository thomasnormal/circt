// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-sim-compile] Functions: 3 total, 0 external, 0 rejected, 3 compilable
// COMPILE: [circt-sim-compile] Demoted 2 intercepted functions to trampolines
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
//
// COMPILED: out=7

func.func @pause_wrapper() -> i32 {
  sim.pause quiet
  %c0 = hw.constant 0 : i32
  return %c0 : i32
}

func.func @terminate_wrapper() -> i32 {
  sim.terminate success, quiet
  %c0 = hw.constant 0 : i32
  return %c0 : i32
}

func.func @keep_alive() -> i32 {
  %c7 = hw.constant 7 : i32
  return %c7 : i32
}

hw.module @top() {
  %c20_i64 = hw.constant 20000000 : i64

  llhd.process {
    %rv = func.call @keep_alive() : () -> i32
    %prefix = sim.fmt.literal "out="
    %nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.dec %rv signed : i32
    %all = sim.fmt.concat (%prefix, %fmt, %nl)
    sim.proc.print %all
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %c20_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
