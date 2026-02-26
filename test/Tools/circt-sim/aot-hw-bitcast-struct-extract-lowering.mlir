// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
// SIM: out=26796
// COMPILED: out=26796

func.func @bitcast_extract_sum(%x: i32) -> i32 {
  %s = hw.bitcast %x : (i32) -> !hw.struct<hi: i16, lo: i16>
  %hi = hw.struct_extract %s["hi"] : !hw.struct<hi: i16, lo: i16>
  %lo = hw.struct_extract %s["lo"] : !hw.struct<hi: i16, lo: i16>
  %hi32 = arith.extui %hi : i16 to i32
  %lo32 = arith.extui %lo : i16 to i32
  %sum = arith.addi %hi32, %lo32 : i32
  return %sum : i32
}

hw.module @test() {
  %in = hw.constant 305419896 : i32  // 0x12345678
  %v = func.call @bitcast_extract_sum(%in) : (i32) -> i32

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
