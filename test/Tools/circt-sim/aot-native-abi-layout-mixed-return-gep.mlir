// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// Regression: mixed native/interpreted object access must use ABI struct
// layout. In this test, the interpreted callee writes an i64 in struct field 1
// of !llvm.struct<(i32, i64)>, which is ABI-aligned at byte offset 8.
// Native caller then reads field 1 through struct GEP. Packed offset rewriting
// (byte offset 4) reads the wrong data.
//
// COMPILE: [circt-compile] Demoted 1 intercepted functions to trampolines
// SIM: read=72623859790382856
// COMPILED: read=72623859790382856

llvm.func @malloc(i64) -> !llvm.ptr

func.func private @"uvm_pkg::mk_pair"() -> !llvm.ptr {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero32 = llvm.mlir.constant(0 : i32) : i32
  %f0 = llvm.mlir.constant(287454020 : i32) : i32
  %f1 = llvm.mlir.constant(72623859790382856 : i64) : i64
  %p = llvm.call @malloc(%size) : (i64) -> !llvm.ptr

  %f0ptr = llvm.getelementptr %p[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i64)>
  llvm.store %f0, %f0ptr : i32, !llvm.ptr

  // Clear struct padding bytes [4..7] so a buggy packed load at +4 is stable.
  %pad = llvm.getelementptr %p[4] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %zero32, %pad : i32, !llvm.ptr

  %f1ptr = llvm.getelementptr %p[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i64)>
  llvm.store %f1, %f1ptr : i64, !llvm.ptr
  return %p : !llvm.ptr
}

func.func private @native_read_pair() -> i64 {
  %p = func.call @"uvm_pkg::mk_pair"() : () -> !llvm.ptr
  %f1ptr = llvm.getelementptr %p[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i64)>
  %v = llvm.load %f1ptr : !llvm.ptr -> i64
  return %v : i64
}

hw.module @top() {
  %prefix = sim.fmt.literal "read="
  %nl = sim.fmt.literal "\0A"
  %t10 = hw.constant 10000000 : i64

  llhd.process {
    %v = func.call @native_read_pair() : () -> i64
    %vf = sim.fmt.dec %v signed : i64
    %msg = sim.fmt.concat (%prefix, %vf, %nl)
    sim.proc.print %msg
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %t10
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
