// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: hw.module @local_var_proc
// CHECK: llhd.process
// CHECK: %[[LOCAL:.*]] = llvm.alloca {{.*}} x i1
// CHECK: llvm.store {{.*}}, %[[LOCAL]]
// CHECK: llvm.load %[[LOCAL]] : !llvm.ptr -> i1
moore.module @local_var_proc(out out: !moore.i1) {
  %out_var = moore.variable : <i1>
  moore.procedure initial {
    %local = moore.variable : <i1>
    %c1 = moore.constant 1 : i1
    moore.blocking_assign %local, %c1 : i1
    %r = moore.read %local : <i1>
    moore.blocking_assign %out_var, %r : i1
    moore.return
  }
  %out_val = moore.read %out_var : <i1>
  moore.output %out_val : !moore.i1
}
