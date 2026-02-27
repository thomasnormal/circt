// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE-NOT: Stripped
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
// SIM: out=1
// COMPILED: out=1

func.func @cmpf_test() -> i32 {
  %a = arith.constant 1.500000e+00 : f32
  %b = arith.constant 2.000000e+00 : f32
  %lt = arith.cmpf olt, %a, %b : f32
  %lt_i32 = arith.extui %lt : i1 to i32
  return %lt_i32 : i32
}

hw.module @test() {
  %v = func.call @cmpf_test() : () -> i32
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
