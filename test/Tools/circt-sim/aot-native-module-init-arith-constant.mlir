// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: env CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=1 circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=NATIVE

// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] Native module init functions: 1
// COMPILE: [circt-compile] Native module init modules: 1 emitted / 1 total
//
// SIM: out=77
//
// NATIVE: [circt-sim] Native module init: top
// NATIVE: out=77

llvm.mlir.global internal @g_counter(0 : i32) : i32

func.func @read_counter() -> i32 {
  %ptr = llvm.mlir.addressof @g_counter : !llvm.ptr
  %v = llvm.load %ptr : !llvm.ptr -> i32
  return %v : i32
}

hw.module @top() {
  %c77 = arith.constant 77 : i32
  %ptr = llvm.mlir.addressof @g_counter : !llvm.ptr
  llvm.store %c77, %ptr : i32, !llvm.ptr

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
