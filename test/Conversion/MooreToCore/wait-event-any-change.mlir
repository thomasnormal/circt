// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: hw.module @m
// CHECK: llhd.process {
// CHECK: llhd.wait ({{.*}} : i1, i1)
moore.module @m() {
  %a = moore.variable : !moore.ref<i1>
  %b = moore.variable : !moore.ref<i1>
  moore.procedure initial {
    moore.wait_event {
      %ra = moore.read %a : !moore.ref<i1>
      %rb = moore.read %b : !moore.ref<i1>
      %and = moore.and %ra, %rb : i1
      moore.detect_event any %and : i1
    }
    moore.return
  }
  moore.output
}
