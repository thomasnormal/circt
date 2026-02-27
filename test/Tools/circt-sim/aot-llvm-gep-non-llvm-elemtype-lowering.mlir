// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE-NOT: Translation to LLVM IR failed
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
// SIM: out=1
// COMPILED: out=1

func.func @gep_hw_struct_elemtype_roundtrip() -> i32 {
  %one64 = llvm.mlir.constant(1 : i64) : i64
  %zero64 = llvm.mlir.constant(0 : i64) : i64
  %one32 = llvm.mlir.constant(1 : i32) : i32
  %zero32 = llvm.mlir.constant(0 : i32) : i32
  %null = llvm.mlir.zero : !llvm.ptr

  %slot = llvm.alloca %one64 x !llvm.array<2 x ptr> : (i64) -> !llvm.ptr
  %elem0 = llvm.getelementptr %slot[%zero64] : (!llvm.ptr, i64) -> !llvm.ptr, !hw.struct<a: i32, b: i32>
  %elem1 = llvm.getelementptr %slot[%one64] : (!llvm.ptr, i64) -> !llvm.ptr, !hw.struct<a: i32, b: i32>

  llvm.store %null, %elem0 : !llvm.ptr, !llvm.ptr
  llvm.store %null, %elem1 : !llvm.ptr, !llvm.ptr

  %v = llvm.load %elem1 : !llvm.ptr -> !llvm.ptr
  %isNull = llvm.icmp "eq" %v, %null : !llvm.ptr
  %out = arith.select %isNull, %one32, %zero32 : i32
  return %out : i32
}

hw.module @top() {
  %prefix = sim.fmt.literal "out="
  %nl = sim.fmt.literal "\0A"
  %t10 = hw.constant 10000000 : i64

  llhd.process {
    %d = llhd.int_to_time %t10
    llhd.wait delay %d, ^print
  ^print:
    %v = func.call @gep_hw_struct_elemtype_roundtrip() : () -> i32
    %vfmt = sim.fmt.dec %v signed : i32
    %msg = sim.fmt.concat (%prefix, %vfmt, %nl)
    sim.proc.print %msg
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
