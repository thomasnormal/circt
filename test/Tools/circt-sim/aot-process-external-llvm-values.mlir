// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s
//
// Ensure process-body extraction accepts common external LLVM values captured
// from the enclosing hw.module region.
//
// CHECK: [circt-compile] Compiled 1 process bodies
// CHECK: [circt-compile] Processes: 1 total, 1 callback-eligible, 0 rejected

llvm.mlir.global internal constant @__packed_hello("h") {addr_space = 0 : i32}

hw.module @top() {
  %false = hw.constant false
  %true = hw.constant true
  %one = llvm.mlir.constant(1 : i64) : i64
  %zero = llvm.mlir.zero : !llvm.ptr
  %undef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  %msg = llvm.mlir.addressof @__packed_hello : !llvm.ptr
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %sig = llhd.sig %false : i1

  llhd.process {
    %slot = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.store %zero, %slot : !llvm.ptr, !llvm.ptr
    %s0 = llvm.insertvalue %msg, %undef[0] : !llvm.struct<(ptr, i64)>
    %s1 = llvm.insertvalue %one, %s0[1] : !llvm.struct<(ptr, i64)>
    %v = llhd.prb %sig : i1
    %nv = comb.xor %v, %true : i1
    llhd.drv %sig, %nv after %eps : i1
    llhd.halt
  }

  hw.output
}
