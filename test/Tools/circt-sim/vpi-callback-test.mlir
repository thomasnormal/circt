// RUN: cc -shared -fPIC -o %t.vpi.so %S/vpi-callback-test.c -ldl
// RUN: circt-sim %s --top cb_test --max-time=1000 --vpi=%t.vpi.so 2>&1 | FileCheck %s

// CHECK: VPI_CB: start_of_simulation
// CHECK: VPI_CB: module={{.*}}
// CHECK: VPI_CB: module_count={{[1-9][0-9]*}}
// CHECK: VPI_CB: signal={{.*}} type={{[0-9]+}} size={{[1-9][0-9]*}}
// CHECK: VPI_CB: time=0:0
// CHECK: VPI_CB: {{[0-9]+}} passed, 0 failed
// CHECK: VPI_CB: end_of_simulation
// CHECK: VPI_CB: FINAL: {{[0-9]+}} passed, 0 failed

hw.module @cb_test(in %clk : i1, in %data_in : i8) {
  %c0 = hw.constant 0 : i16
  %wide = llhd.sig %c0 : i16
}
