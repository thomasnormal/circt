// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE-NOT: Stripped
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
// SIM: out=20
// COMPILED: out=20

func.func @sum_pair() -> i32 {
  %hi = hw.constant 18 : i16
  %lo = hw.constant 2 : i16
  %pair = hw.struct_create (%hi, %lo) : !hw.struct<hi: i16, lo: i16>
  %llvm_pair = builtin.unrealized_conversion_cast %pair : !hw.struct<hi: i16, lo: i16> to !llvm.struct<(i16, i16)>
  %a = llvm.extractvalue %llvm_pair[0] : !llvm.struct<(i16, i16)>
  %b = llvm.extractvalue %llvm_pair[1] : !llvm.struct<(i16, i16)>
  %a32 = arith.extui %a : i16 to i32
  %b32 = arith.extui %b : i16 to i32
  %sum = arith.addi %a32, %b32 : i32
  return %sum : i32
}

hw.module @test() {
  %v = func.call @sum_pair() : () -> i32
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
