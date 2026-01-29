// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @test_bmc_finish() -> i1
// CHECK: scf.for {{.*}} iter_args({{.*}}, [[VIOLATED:%.+]] = {{.+}})
// CHECK: smt.not
// CHECK: smt.assert
func.func @test_bmc_finish() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%arg0: i1):
    %true = hw.constant true
    %false = hw.constant false
    verif.assume %true {bmc.finish} : i1
    verif.assert %false : i1
    verif.yield %arg0 : i1
  }
  func.return %bmc : i1
}
