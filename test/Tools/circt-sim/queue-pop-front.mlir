// RUN: circt-sim %s | FileCheck %s

// Test queue push_back + pop_front_ptr operations.
// Previously, pop_front_ptr crashed with SIGSEGV because it used
// reinterpret_cast on a simulated address instead of looking up
// the data through findMemoryBlockByAddress.

// CHECK: popped=42

llvm.func @__moore_queue_create(i64) -> !llvm.ptr
llvm.func @__moore_queue_push_back(!llvm.ptr, !llvm.ptr, i64)
llvm.func @__moore_queue_pop_front_ptr(!llvm.ptr, !llvm.ptr, i64)

hw.module @test() {
  llhd.process {
    %one = llvm.mlir.constant(1 : i64) : i64
    %four = llvm.mlir.constant(4 : i64) : i64

    // Create a queue with 4-byte elements
    %queue_raw = llvm.call @__moore_queue_create(%four) : (i64) -> !llvm.ptr

    // Allocate the queue struct (ptr + len = 16 bytes) and store
    %queue_struct = llvm.alloca %one x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
    %c0_i64 = llvm.mlir.constant(0 : i64) : i64
    %null = llvm.mlir.zero : !llvm.ptr
    // Initialize: data=null, len=0
    %zero_struct = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %s1 = llvm.insertvalue %null, %zero_struct[0] : !llvm.struct<(ptr, i64)>
    %s2 = llvm.insertvalue %c0_i64, %s1[1] : !llvm.struct<(ptr, i64)>
    llvm.store %s2, %queue_struct : !llvm.struct<(ptr, i64)>, !llvm.ptr

    // Push value 42
    %c42 = hw.constant 42 : i32
    %elem_alloca = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    llvm.store %c42, %elem_alloca : i32, !llvm.ptr
    llvm.call @__moore_queue_push_back(%queue_struct, %elem_alloca, %four) : (!llvm.ptr, !llvm.ptr, i64) -> ()

    // Pop front into result
    %result_alloca = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    %c0_i32 = hw.constant 0 : i32
    llvm.store %c0_i32, %result_alloca : i32, !llvm.ptr
    llvm.call @__moore_queue_pop_front_ptr(%queue_struct, %result_alloca, %four) : (!llvm.ptr, !llvm.ptr, i64) -> ()

    // Read result
    %result = llvm.load %result_alloca : !llvm.ptr -> i32

    %fmt_prefix = sim.fmt.literal "popped="
    %fmt_nl = sim.fmt.literal "\0A"
    %fmt_val = sim.fmt.dec %result : i32
    %out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
