// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Deferred immediate assertions lowered in an `always` procedure must not
// become tight zero-delay loops. They need an explicit wait between
// activations so simulation does not spin in one activation.

// CHECK-LABEL: hw.module @DeferredAlways
// CHECK: llhd.process
// CHECK: verif.assume
// CHECK: llhd.wait
moore.module @DeferredAlways(in %c : !moore.l1) {
  moore.procedure always {
    moore.assume observed %c : l1
    moore.return
  }
}

