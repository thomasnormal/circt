// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Wait Condition Operation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_wait_condition
moore.module @test_wait_condition(in %clk: !moore.i1) {
  // CHECK: llhd.process
  moore.procedure initial {
    // CHECK: cf.br ^[[CHECK:.*]]
    // CHECK: ^[[CHECK]]:
    // CHECK: cf.cond_br %{{.*}}, ^[[RESUME:.*]], ^[[WAIT:.*]]
    // CHECK: ^[[WAIT]]:
    // CHECK: llhd.wait (%{{.*}} : i1), ^[[CHECK]]
    // CHECK: ^[[RESUME]]:
    moore.wait_condition %clk : !moore.i1
    moore.return
  }
  moore.output
}
