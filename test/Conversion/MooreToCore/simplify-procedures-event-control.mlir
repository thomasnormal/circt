// RUN: circt-opt --moore-simplify-procedures --convert-moore-to-core %s | FileCheck %s

moore.module @M() {
  %clk = moore.variable : <i1>
  %y = moore.variable : <i1>

  moore.procedure always {
    moore.wait_event {
      %r = moore.read %clk : <i1>
      moore.detect_event posedge %r : i1
    }
    %v = moore.constant 1 : i1
    moore.blocking_assign %y, %v : i1
    moore.return
  }
}

// CHECK-LABEL: hw.module @M() {
// CHECK: %[[CLK:.+]] = llhd.sig
// CHECK: %[[PREV:.+]] = llhd.prb %[[CLK]]
// CHECK: llhd.wait (%[[PREV]]
// CHECK: %[[CUR:.+]] = llhd.prb %[[CLK]]
// CHECK-NOT: llvm.alloca
