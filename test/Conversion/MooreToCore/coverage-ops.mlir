// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test for MooreToCore lowering of coverage operations.
// These operations are lowered to runtime function calls that implement
// functional coverage tracking.

// CHECK-DAG: llvm.mlir.global internal @__cg_handle_TestCG
// CHECK-DAG: llvm.mlir.global{{.*}}constant @__cg_name_TestCG("TestCG
// CHECK-DAG: llvm.mlir.global{{.*}}constant @__cp_name_TestCG_data("data
// CHECK-DAG: llvm.mlir.global{{.*}}constant @__cp_name_TestCG_addr("addr
// CHECK-DAG: llvm.func @__moore_covergroup_create(!llvm.ptr, i32) -> !llvm.ptr
// CHECK-DAG: llvm.func @__moore_coverpoint_init(!llvm.ptr, i32, !llvm.ptr)
// CHECK-DAG: llvm.func @__moore_coverpoint_sample(!llvm.ptr, i32, i64)
// CHECK-DAG: llvm.func @__moore_covergroup_get_coverage(!llvm.ptr) -> f64
// CHECK-DAG: llvm.func @__moore_cross_create(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK-DAG: llvm.func @__moore_cross_sample(!llvm.ptr, !llvm.ptr, i32)

// Covergroup declaration with two coverpoints
moore.covergroup.decl @TestCG {
  moore.coverpoint.decl @data : !moore.i8 {}
  moore.coverpoint.decl @addr : !moore.i16 {}
}

// CHECK-LABEL: func @TestCovergroupInst
func.func @TestCovergroupInst() -> !moore.covergroup<@TestCG> {
  // CHECK: llvm.call @__cg_init_TestCG() : () -> ()
  // CHECK: [[HANDLE_PTR:%.+]] = llvm.mlir.addressof @__cg_handle_TestCG : !llvm.ptr
  // CHECK: [[HANDLE:%.+]] = llvm.load [[HANDLE_PTR]] : !llvm.ptr -> !llvm.ptr
  // CHECK: return [[HANDLE]] : !llvm.ptr
  %cg = moore.covergroup.inst @TestCG : !moore.covergroup<@TestCG>
  return %cg : !moore.covergroup<@TestCG>
}

// CHECK-LABEL: func @TestCovergroupSample
// CHECK-SAME: (%[[CG:.*]]: !llvm.ptr, %[[DATA:.*]]: i8, %[[ADDR:.*]]: i16)
func.func @TestCovergroupSample(%cg: !moore.covergroup<@TestCG>, %data: !moore.i8, %addr: !moore.i16) {
  // CHECK: %[[DATA_EXT:.*]] = arith.extui %[[DATA]] : i8 to i64
  // CHECK: %[[IDX0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.call @__moore_coverpoint_sample(%[[CG]], %[[IDX0]], %[[DATA_EXT]]) : (!llvm.ptr, i32, i64) -> ()
  // CHECK: %[[ADDR_EXT:.*]] = arith.extui %[[ADDR]] : i16 to i64
  // CHECK: %[[IDX1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.call @__moore_coverpoint_sample(%[[CG]], %[[IDX1]], %[[ADDR_EXT]]) : (!llvm.ptr, i32, i64) -> ()
  moore.covergroup.sample %cg(%data, %addr) : !moore.covergroup<@TestCG> (!moore.i8, !moore.i16)
  return
}

// CHECK-LABEL: func @TestCovergroupGetCoverage
// CHECK-SAME: (%[[CG:.*]]: !llvm.ptr)
func.func @TestCovergroupGetCoverage(%cg: !moore.covergroup<@TestCG>) -> !moore.f64 {
  // CHECK: %[[COV:.*]] = llvm.call @__moore_covergroup_get_coverage(%[[CG]]) : (!llvm.ptr) -> f64
  // CHECK: return %[[COV]] : f64
  %cov = moore.covergroup.get_coverage %cg : !moore.covergroup<@TestCG> -> !moore.f64
  return %cov : !moore.f64
}

// Test covergroup with cross coverage (now fully lowered)
// Function declarations are checked at the top of this file.
moore.covergroup.decl @CrossCG {
  moore.coverpoint.decl @a : !moore.i4 {}
  moore.coverpoint.decl @b : !moore.i4 {}
  moore.covercross.decl @ab targets [@a, @b] {
  }
}

// CHECK-LABEL: func @TestCrossInst
func.func @TestCrossInst() -> !moore.covergroup<@CrossCG> {
  // Cross coverage is now initialized with __moore_cross_create
  // CHECK: llvm.call @__cg_init_CrossCG() : () -> ()
  %cg = moore.covergroup.inst @CrossCG : !moore.covergroup<@CrossCG>
  return %cg : !moore.covergroup<@CrossCG>
}
