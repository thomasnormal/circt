// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED
// RUN: env CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=1 circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=NATIVE

// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] Native module init functions: 1
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
//
// SIM: out=12345
//
// COMPILED: out=12345
//
// NATIVE: [circt-sim] Native module init: top
// NATIVE: out=0

llvm.func @memset(!llvm.ptr, i8, i64) -> !llvm.ptr
llvm.mlir.global internal @g_counter(12345 : i32) : i32

func.func @read_counter() -> i32 {
  %ptr = llvm.mlir.addressof @g_counter : !llvm.ptr
  %v = llvm.load %ptr : !llvm.ptr -> i32
  return %v : i32
}

hw.module @top() {
  %ptr = llvm.mlir.addressof @g_counter : !llvm.ptr
  %c0_i8 = llvm.mlir.constant(0 : i8) : i8
  %c4_i64 = llvm.mlir.constant(4 : i64) : i64
  %ignored = llvm.call @memset(%ptr, %c0_i8, %c4_i64) : (!llvm.ptr, i8, i64) -> !llvm.ptr

  %fmtPrefix = sim.fmt.literal "out="
  %fmtNl = sim.fmt.literal "\0A"
  %c5_i64 = hw.constant 5000000 : i64
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^print
  ^print:
    %r = func.call @read_counter() : () -> i32
    %fmtV = sim.fmt.dec %r signed : i32
    %fmtOut = sim.fmt.concat (%fmtPrefix, %fmtV, %fmtNl)
    sim.proc.print %fmtOut
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %c10_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
