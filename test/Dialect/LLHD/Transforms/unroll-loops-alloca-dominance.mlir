// RUN: circt-opt --verify-each --llhd-unroll-loops %s | FileCheck %s

llvm.func @sink_ptr(!llvm.ptr)

// CHECK-LABEL: hw.module @HoistAllocaCountConstant
hw.module @HoistAllocaCountConstant() {
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c2_i32 = hw.constant 2 : i32
  llhd.combinational {
    // The hoisted alloca count operand must dominate the alloca.
    // CHECK: llhd.combinational
    // CHECK: %[[COUNT:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT: %[[PTR:.*]] = llvm.alloca %[[COUNT]] x i32 : (i64) -> !llvm.ptr
    cf.br ^header(%c0_i32 : i32)
  ^header(%i: i32):  // 2 preds: ^bb0, ^body
    %cond = comb.icmp slt %i, %c2_i32 : i32
    cf.cond_br %cond, ^body, ^exit
  ^body:  // pred: ^header
    %count = llvm.mlir.constant(1 : i64) : i64
    %ptr = llvm.alloca %count x i32 : (i64) -> !llvm.ptr
    llvm.call @sink_ptr(%ptr) : (!llvm.ptr) -> ()
    %ip = comb.add %i, %c1_i32 : i32
    cf.br ^header(%ip : i32)
  ^exit:  // pred: ^header
    llhd.yield
  }
}
