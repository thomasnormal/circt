// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE-NOT: Stripped
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
// SIM: out=11
// COMPILED: out=11

func.func @select_value(%cond: i1) -> i32 {
  %c11 = hw.constant 11 : i32
  %c22 = hw.constant 22 : i32
  %c0 = hw.constant 0 : i32
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %s1 = hw.struct_create (%c11, %c0) : !hw.struct<value: i32, unknown: i32>
  cf.br ^merge(%s1 : !hw.struct<value: i32, unknown: i32>)
^bb2:
  %s2 = hw.struct_create (%c22, %c0) : !hw.struct<value: i32, unknown: i32>
  cf.br ^merge(%s2 : !hw.struct<value: i32, unknown: i32>)
^merge(%s: !hw.struct<value: i32, unknown: i32>):
  %v = hw.struct_extract %s["value"] : !hw.struct<value: i32, unknown: i32>
  return %v : i32
}

hw.module @test() {
  %true = hw.constant true
  %v = func.call @select_value(%true) : (i1) -> i32
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
