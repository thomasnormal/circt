// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: hw.module @top
// CHECK: llhd.process
// CHECK: llhd.wait
// CHECK-NOT: llhd.wait (
moore.module @top() {
  %c = moore.variable : !moore.ref<class<@C>>
  %result = moore.variable : !moore.ref<i1>
  moore.procedure always_comb {
    %cval = moore.read %c : !moore.ref<class<@C>>
    %null = moore.class.null : !moore.class<@C>
    %cmp = moore.class_handle_cmp ne %cval, %null : !moore.class<@C> -> i1
    // Write the result to prevent DCE
    moore.blocking_assign %result, %cmp : i1
    moore.return
  }
  moore.output
}

moore.class.classdecl @C {
}
