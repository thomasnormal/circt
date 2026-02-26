// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE-NOT: Stripped
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
// SIM: out=3
// COMPILED: out=3

func.func @fptosi_test() -> i32 {
  %x = arith.constant 3.750000e+00 : f32
  %i = arith.fptosi %x : f32 to i32
  return %i : i32
}

hw.module @test() {
  %v = func.call @fptosi_test() : () -> i32
  %prefix = sim.fmt.literal "out="
  %newline = sim.fmt.literal "\0A"
  %vfmt = sim.fmt.dec %v : i32
  %msg = sim.fmt.concat (%prefix, %vfmt, %newline)

  %t10 = hw.constant 10000000 : i64
  llhd.process {
    %d = llhd.int_to_time %t10
    llhd.wait delay %d, ^print
  ^print:
    sim.proc.print %msg
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
