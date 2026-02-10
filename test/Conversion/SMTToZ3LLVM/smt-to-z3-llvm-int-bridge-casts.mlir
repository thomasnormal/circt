// RUN: circt-opt %s --lower-smt-to-z3-llvm --reconcile-unrealized-casts | FileCheck %s

func.func @test_bridge(%arg0: i1, %arg1: i8) {
  smt.solver(%arg0, %arg1) : (i1, i8) -> () {
  ^bb0(%b: i1, %x: i8):
    %b_bv = builtin.unrealized_conversion_cast %b : i1 to !smt.bv<1>
    %x_bv = builtin.unrealized_conversion_cast %x : i8 to !smt.bv<8>
    %eq1 = smt.eq %b_bv, %b_bv : !smt.bv<1>
    %eq2 = smt.eq %x_bv, %x_bv : !smt.bv<8>
    smt.assert %eq1
    smt.assert %eq2
    smt.check sat {
      smt.yield
    } unknown {
      smt.yield
    } unsat {
      smt.yield
    }
    smt.yield
  }
  return
}

// CHECK-LABEL: llvm.func @solver_0(%arg0: i1, %arg1: i8)
// CHECK: llvm.zext %arg0 : i1 to i64
// CHECK: llvm.call @Z3_mk_unsigned_int64({{.*}}) : (!llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr
// CHECK: llvm.zext %arg1 : i8 to i64
// CHECK: llvm.call @Z3_mk_unsigned_int64({{.*}}) : (!llvm.ptr, i64, !llvm.ptr) -> !llvm.ptr
// CHECK-NOT: unrealized_conversion_cast
