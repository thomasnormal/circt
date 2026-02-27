// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE-NOT: Stripped
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
// SIM: out=5
// COMPILED: out=5

func.func @mix_struct_fields() -> i32 {
  %f = arith.constant 3.250000e+00 : f32
  %i = hw.constant 2 : i32
  %s = hw.struct_create (%f, %i) : !hw.struct<f: f32, i: i32>
  %ls = builtin.unrealized_conversion_cast %s : !hw.struct<f: f32, i: i32> to !llvm.struct<(f32, i32)>
  %f_field = llvm.extractvalue %ls[0] : !llvm.struct<(f32, i32)>
  %i_field = llvm.extractvalue %ls[1] : !llvm.struct<(f32, i32)>
  %f_i32 = arith.fptosi %f_field : f32 to i32
  %sum = arith.addi %f_i32, %i_field : i32
  return %sum : i32
}

hw.module @test() {
  %v = func.call @mix_struct_fields() : () -> i32
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
