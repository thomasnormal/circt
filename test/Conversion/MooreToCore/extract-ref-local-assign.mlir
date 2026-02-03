// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: hw.module @extract_ref_local_assign
// CHECK: llhd.process
// CHECK: %[[BASE:.*]] = llvm.alloca {{.*}} x !llvm.struct<(i8, i8)>
// CHECK: llvm.store {{.*}}, %[[BASE]]
// CHECK: llvm.load %[[BASE]]
// CHECK: llvm.store {{.*}}, %[[BASE]]
moore.module @extract_ref_local_assign(out out: !moore.l8) {
  %out_var = moore.variable : <l8>
  moore.procedure initial {
    %vec = moore.variable : <l8>
    %bit_ref = moore.extract_ref %vec from 3 : !moore.ref<l8> -> !moore.ref<l1>
    %val = moore.constant 1 : l1
    moore.blocking_assign %bit_ref, %val : l1
    %read = moore.read %vec : <l8>
    moore.blocking_assign %out_var, %read : l8
    moore.return
  }
  %out_val = moore.read %out_var : <l8>
  moore.output %out_val : !moore.l8
}
