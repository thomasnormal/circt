// RUN: circt-sim %s | FileCheck %s

// CHECK: assoc_val=99

llvm.func @__moore_assoc_create(i32, i32) -> !llvm.ptr
llvm.func @__moore_assoc_get_ref(!llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr

hw.module @test() {
  %fmt_prefix = sim.fmt.literal "assoc_val="
  %fmt_nl = sim.fmt.literal "\0A"
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %c4_i32 = llvm.mlir.constant(4 : i32) : i32
  %c42_i32 = llvm.mlir.constant(42 : i32) : i32
  %c99_i32 = llvm.mlir.constant(99 : i32) : i32

  llhd.process {
    %key_mem = llvm.alloca %c1_i64 x i32 : (i64) -> !llvm.ptr
    llvm.store %c42_i32, %key_mem : i32, !llvm.ptr

    %array = llvm.call @__moore_assoc_create(%c4_i32, %c4_i32) : (i32, i32) -> !llvm.ptr
    %val_ptr = llvm.call @__moore_assoc_get_ref(%array, %key_mem, %c4_i32) : (!llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr

    llvm.store %c99_i32, %val_ptr : i32, !llvm.ptr
    %loaded = llvm.load %val_ptr : !llvm.ptr -> i32

    %fmt_val = sim.fmt.dec %loaded : i32
    %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
