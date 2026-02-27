// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

module {
  moore.module @AlwaysCombLatchProcedures() {
    %comb_input = moore.variable : !moore.ref<i1>

    moore.procedure always_comb {
      %0 = moore.read %comb_input : !moore.ref<i1>
      func.call @dummyA() : () -> ()
      moore.return
    }

    moore.procedure always_latch {
      %0 = moore.read %comb_input : !moore.ref<i1>
      func.call @dummyA() : () -> ()
      moore.return
    }

    moore.output
  }

  func.func private @dummyA() -> ()
}

// CHECK-LABEL: hw.module @AlwaysCombLatchProcedures
// CHECK: %comb_input = llhd.sig

// CHECK: llhd.process {
// CHECK:   cf.br ^[[AC_BODY:.+]]
// CHECK: ^[[AC_BODY]]:
// CHECK:   {{.*}} = llhd.prb %comb_input
// CHECK:   func.call @dummyA()
// CHECK:   cf.br ^[[AC_WAIT:.+]]
// CHECK: ^[[AC_WAIT]]:
// CHECK:   llhd.wait
// CHECK:   ^[[AC_BODY]]
// CHECK: }

// CHECK: llhd.process {
// CHECK:   cf.br ^[[AL_BODY:.+]]
// CHECK: ^[[AL_BODY]]:
// CHECK:   {{.*}} = llhd.prb %comb_input
// CHECK:   func.call @dummyA()
// CHECK:   cf.br ^[[AL_WAIT:.+]]
// CHECK: ^[[AL_WAIT]]:
// CHECK:   llhd.wait
// CHECK:   ^[[AL_BODY]]
// CHECK: }
