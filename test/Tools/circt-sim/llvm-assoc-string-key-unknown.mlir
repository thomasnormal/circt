// RUN: circt-sim %s | FileCheck %s

// CHECK: exists=0

llvm.func @__moore_assoc_create(i32, i32) -> !llvm.ptr
llvm.func @__moore_assoc_exists(!llvm.ptr, !llvm.ptr) -> i32

hw.module @test() {
  %fmt_prefix = sim.fmt.literal "exists="
  %fmt_nl = sim.fmt.literal "\0A"
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %c0_i32 = llvm.mlir.constant(0 : i32) : i32
  %c4_i32 = llvm.mlir.constant(4 : i32) : i32
  %c4_i64 = llvm.mlir.constant(4 : i64) : i64
  %c65536_i64 = llvm.mlir.constant(65536 : i64) : i64
  %undef_struct = llvm.mlir.undef : !llvm.struct<(ptr, i64)>

  llhd.process {
    %array = llvm.call @__moore_assoc_create(%c0_i32, %c4_i32) : (i32, i32) -> !llvm.ptr

    %key_mem = llvm.alloca %c1_i64 x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
    %str_ptr = llvm.inttoptr %c65536_i64 : i64 to !llvm.ptr
    %key_0 = llvm.insertvalue %str_ptr, %undef_struct[0] : !llvm.struct<(ptr, i64)>
    %key_1 = llvm.insertvalue %c4_i64, %key_0[1] : !llvm.struct<(ptr, i64)>
    llvm.store %key_1, %key_mem : !llvm.struct<(ptr, i64)>, !llvm.ptr

    %exists = llvm.call @__moore_assoc_exists(%array, %key_mem) : (!llvm.ptr, !llvm.ptr) -> i32
    %fmt_val = sim.fmt.dec %exists : i32
    %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
