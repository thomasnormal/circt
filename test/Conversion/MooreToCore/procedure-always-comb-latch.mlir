// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

moore.module @AlwaysCombAndLatch() {
  moore.procedure always_comb {
    func.call @dummy() : () -> ()
    moore.return
  }

  moore.procedure always_latch {
    func.call @dummy() : () -> ()
    moore.return
  }
}

func.func private @dummy()

// CHECK-LABEL: hw.module @AlwaysCombAndLatch
// CHECK: llhd.process {
// CHECK:   cf.br ^[[COMB_BODY:.+]]
// CHECK: ^[[COMB_BODY]]:
// CHECK:   func.call @dummy()
// CHECK:   cf.br ^[[COMB_WAIT:.+]]
// CHECK: ^[[COMB_WAIT]]:
// CHECK:   llhd.wait ^[[COMB_BODY]]
// CHECK: }
// CHECK: llhd.process {
// CHECK:   cf.br ^[[LATCH_BODY:.+]]
// CHECK: ^[[LATCH_BODY]]:
// CHECK:   func.call @dummy()
// CHECK:   cf.br ^[[LATCH_WAIT:.+]]
// CHECK: ^[[LATCH_WAIT]]:
// CHECK:   llhd.wait ^[[LATCH_BODY]]
// CHECK: }
