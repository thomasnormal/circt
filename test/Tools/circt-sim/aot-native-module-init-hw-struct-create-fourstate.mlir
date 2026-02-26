// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: env CIRCT_AOT_ENABLE_NATIVE_MODULE_INIT=1 circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=NATIVE

// COMPILE: [circt-sim-compile] Native module init functions: 1
// COMPILE: [circt-sim-compile] Native module init modules: 1 emitted / 1 total
// COMPILE-NOT: unsupported_op:hw.struct_create
//
// SIM: pair=-1,0
//
// NATIVE: [circt-sim] Native module init: top
// NATIVE: pair=-1,0

llvm.mlir.global internal @g_val(0 : i1) : i1
llvm.mlir.global internal @g_unk(0 : i1) : i1

func.func @read_val() -> i1 {
  %ptr = llvm.mlir.addressof @g_val : !llvm.ptr
  %v = llvm.load %ptr : !llvm.ptr -> i1
  return %v : i1
}

func.func @read_unk() -> i1 {
  %ptr = llvm.mlir.addressof @g_unk : !llvm.ptr
  %u = llvm.load %ptr : !llvm.ptr -> i1
  return %u : i1
}

hw.module @top() {
  %valPtr = llvm.mlir.addressof @g_val : !llvm.ptr
  %unkPtr = llvm.mlir.addressof @g_unk : !llvm.ptr
  %t = arith.constant true
  %f = arith.constant false
  %sel = hw.struct_create (%t, %f) : !hw.struct<value: i1, unknown: i1>
  %value = hw.struct_extract %sel["value"] : !hw.struct<value: i1, unknown: i1>
  %unknown = hw.struct_extract %sel["unknown"] : !hw.struct<value: i1, unknown: i1>
  llvm.store %value, %valPtr : i1, !llvm.ptr
  llvm.store %unknown, %unkPtr : i1, !llvm.ptr

  %fmtPair = sim.fmt.literal "pair="
  %fmtComma = sim.fmt.literal ","
  %fmtNL = sim.fmt.literal "\0A"
  %c5_i64 = hw.constant 5000000 : i64
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^print
  ^print:
    %v = func.call @read_val() : () -> i1
    %u = func.call @read_unk() : () -> i1
    %fmtV = sim.fmt.dec %v signed : i1
    %fmtU = sim.fmt.dec %u signed : i1
    %fmtOut = sim.fmt.concat (%fmtPair, %fmtV, %fmtComma, %fmtU, %fmtNL)
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
