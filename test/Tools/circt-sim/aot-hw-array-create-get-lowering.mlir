// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
//
// SIM: out=20
//
// COMPILED: out=20

func.func @pick_array_elem() -> i32 {
  %c10 = hw.constant 10 : i32
  %c20 = hw.constant 20 : i32
  %c30 = hw.constant 30 : i32
  %c40 = hw.constant 40 : i32
  %idx = hw.constant 2 : i2
  %arr = hw.array_create %c10, %c20, %c30, %c40 : i32
  %elem = hw.array_get %arr[%idx] : !hw.array<4xi32>, i2
  return %elem : i32
}

hw.module @top() {
  llhd.process {
    %x = func.call @pick_array_elem() : () -> i32
    %prefix = sim.fmt.literal "out="
    %nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.dec %x signed : i32
    %all = sim.fmt.concat (%prefix, %fmt, %nl)
    sim.proc.print %all
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
