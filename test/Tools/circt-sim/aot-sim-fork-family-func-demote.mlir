// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] Demoted 1 intercepted functions to trampolines
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
//
// COMPILED: out=9

func.func @fork_wait_wrapper() -> i32 {
  %handle = sim.fork join_type "join_none" {
    sim.fork.terminator
  }, {
    sim.fork.terminator
  }
  sim.wait_fork
  %c9 = hw.constant 9 : i32
  return %c9 : i32
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}

hw.module @top() {
  %c20_i64 = hw.constant 20000000 : i64

  llhd.process {
    %rv0 = func.call @fork_wait_wrapper() : () -> i32
    %rv = func.call @keep_alive(%rv0) : (i32) -> i32
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
