// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test that ignore_asserts_until attribute gates property checking until
// a certain number of iterations have passed.

// CHECK-DAG: arith.constant 3
// CHECK-DAG: arith.constant true
// CHECK-DAG: arith.constant false
// CHECK: scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args({{%.+}} = {{%.+}}, {{%.+}} = {{%.+}})
// Loop is called first
// CHECK: func.call @bmc_loop
// Circuit returns outputs + property value (!smt.bv<1>)
// CHECK: func.call @bmc_circuit
// CHECK-SAME: -> (!smt.bv<32>, !smt.bv<1>)
// ignore_asserts_until check (compare iteration with threshold)
// CHECK: arith.cmpi ult
// CHECK: scf.if
// Skip checking if before threshold
// CHECK:     scf.yield
// CHECK: } else {
// Check the property
// CHECK:     smt.not
// CHECK:     smt.push 1
// CHECK:     smt.assert
// CHECK:     smt.check sat
// CHECK:     smt.pop 1
// CHECK:     arith.ori
// CHECK:     scf.yield
// CHECK: }
// CHECK: scf.yield


func.func @test_bmc() -> (i1) {
  %bmc = verif.bmc bound 10 num_regs 0 initial_values [] attributes {ignore_asserts_until = 3 : i64}
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%arg0: i32):
    %true = hw.constant true
    verif.assert %true : i1
    verif.yield %arg0 : i32
  }
  func.return %bmc : i1
}
