// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-compile] Functions: 3 total, 0 external, 0 rejected, 3 compilable
// COMPILE-NOT: Stripped
// COMPILE: [circt-compile] 3 functions + 0 processes ready for codegen
// SIM: out=12
// COMPILED: out=12

func.func @pack4(%v: i16, %u: i16) -> !hw.struct<value: i16, unknown: i16> {
  %s = hw.struct_create (%v, %u) : !hw.struct<value: i16, unknown: i16>
  return %s : !hw.struct<value: i16, unknown: i16>
}

func.func @sum4(%s: !hw.struct<value: i16, unknown: i16>) -> i32 {
  %v = hw.struct_extract %s["value"] : !hw.struct<value: i16, unknown: i16>
  %u = hw.struct_extract %s["unknown"] : !hw.struct<value: i16, unknown: i16>
  %v32 = arith.extui %v : i16 to i32
  %u32 = arith.extui %u : i16 to i32
  %sum = arith.addi %v32, %u32 : i32
  return %sum : i32
}

func.func @driver() -> i32 {
  %v = hw.constant 9 : i16
  %u = hw.constant 3 : i16
  %s = func.call @pack4(%v, %u) : (i16, i16) -> !hw.struct<value: i16, unknown: i16>
  %sum = func.call @sum4(%s) : (!hw.struct<value: i16, unknown: i16>) -> i32
  return %sum : i32
}

hw.module @test() {
  %v = func.call @driver() : () -> i32
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
