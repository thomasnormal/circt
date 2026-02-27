// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
//
// SIM: len_after=100001
//
// COMPILED: len_after=100001

llvm.func @__moore_queue_pop_front_ptr(!llvm.ptr, !llvm.ptr, i64)

func.func @pop_bad_queue_len() -> i64 {
  %one = llvm.mlir.constant(1 : i64) : i64
  %four = llvm.mlir.constant(4 : i64) : i64
  %bad_len = llvm.mlir.constant(100001 : i64) : i64

  %queue_struct = llvm.alloca %one x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
  %dummy_data = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
  %result = llvm.alloca %one x i32 : (i64) -> !llvm.ptr

  %undef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  %s0 = llvm.insertvalue %dummy_data, %undef[0] : !llvm.struct<(ptr, i64)>
  %s1 = llvm.insertvalue %bad_len, %s0[1] : !llvm.struct<(ptr, i64)>
  llvm.store %s1, %queue_struct : !llvm.struct<(ptr, i64)>, !llvm.ptr

  // Must be ignored due to invalid queue length.
  llvm.call @__moore_queue_pop_front_ptr(%queue_struct, %result, %four) : (!llvm.ptr, !llvm.ptr, i64) -> ()

  %after = llvm.load %queue_struct : !llvm.ptr -> !llvm.struct<(ptr, i64)>
  %len_after = llvm.extractvalue %after[1] : !llvm.struct<(ptr, i64)>
  return %len_after : i64
}

hw.module @top() {
  llhd.process {
    %len_after = func.call @pop_bad_queue_len() : () -> i64
    %fmt_prefix = sim.fmt.literal "len_after="
    %fmt_nl = sim.fmt.literal "\0A"
    %fmt_val = sim.fmt.dec %len_after : i64
    %out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %out
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
