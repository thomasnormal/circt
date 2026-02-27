// RUN: circt-compile --emit-llvm %s -o %t.ll 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: FileCheck %s --check-prefix=LLVM < %t.ll
// XFAIL: *

// Regression: trampoline ABI packing for integers wider than 64 bits must
// preserve all bits across slot packing/unpacking.
//
// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] Generated 1 interpreter trampolines
// COMPILE: [circt-compile] Wrote LLVM IR to
//
// LLVM: %[[ARG_LO:[0-9]+]] = trunc i128 %{{[0-9]+}} to i64
// LLVM: %[[ARG_SHIFT:[0-9]+]] = lshr i128 %{{[0-9]+}}, 64
// LLVM: %[[ARG_HI:[0-9]+]] = trunc{{.*}} i128 %[[ARG_SHIFT]] to i64
// LLVM: call void @__circt_sim_call_interpreted(ptr {{.*}}, i32 0, ptr {{.*}}, i32 2, ptr {{.*}}, i32 2)
// LLVM: %[[RET_LO_LOAD:[0-9]+]] = load i64, ptr {{.*}}
// LLVM: %[[RET_LO_EXT:[0-9]+]] = zext i64 %[[RET_LO_LOAD]] to i128
// LLVM: %[[RET_HI_LOAD:[0-9]+]] = load i64, ptr {{.*}}
// LLVM: %[[RET_HI_EXT:[0-9]+]] = zext i64 %[[RET_HI_LOAD]] to i128
// LLVM: %[[RET_HI_SHIFT:[0-9]+]] = shl{{.*}} i128 %[[RET_HI_EXT]], 64
// LLVM: %[[RET_JOIN:[0-9]+]] = or{{.*}} i128 %[[RET_HI_SHIFT]], %[[RET_LO_EXT]]
// LLVM: store i128 %[[RET_JOIN]], ptr %0

func.func @entry(%x: i128) -> i128 {
  %r = llvm.call @ext_i128(%x) : (i128) -> i128
  return %r : i128
}

llvm.func @ext_i128(i128) -> i128
