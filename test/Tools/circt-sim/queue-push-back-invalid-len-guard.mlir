// RUN: circt-sim %s | FileCheck %s

// REPRO (pre-fix):
// __moore_queue_push_back accepted invalid queue headers and attempted to
// allocate/copy using corrupted lengths. This test sets an out-of-range queue
// length and checks that push_back is ignored.

// CHECK: len_after=100001

llvm.func @__moore_queue_push_back(!llvm.ptr, !llvm.ptr, i64)

hw.module @test() {
  llhd.process {
    %one = llvm.mlir.constant(1 : i64) : i64
    %four = llvm.mlir.constant(4 : i64) : i64
    %bad_len = llvm.mlir.constant(100001 : i64) : i64
    %elem = llvm.mlir.constant(123 : i32) : i32

    %queue_struct = llvm.alloca %one x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
    %dummy_data = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    %elem_alloca = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    llvm.store %elem, %elem_alloca : i32, !llvm.ptr

    %undef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %s0 = llvm.insertvalue %dummy_data, %undef[0] : !llvm.struct<(ptr, i64)>
    %s1 = llvm.insertvalue %bad_len, %s0[1] : !llvm.struct<(ptr, i64)>
    llvm.store %s1, %queue_struct : !llvm.struct<(ptr, i64)>, !llvm.ptr

    // Must be ignored due to invalid queue length.
    llvm.call @__moore_queue_push_back(%queue_struct, %elem_alloca, %four) : (!llvm.ptr, !llvm.ptr, i64) -> ()

    %after = llvm.load %queue_struct : !llvm.ptr -> !llvm.struct<(ptr, i64)>
    %len_after = llvm.extractvalue %after[1] : !llvm.struct<(ptr, i64)>

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
