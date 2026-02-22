// RUN: cc -shared -fPIC -o %t.vpi.so %S/vpi-basic-test.c -ldl
// RUN: circt-sim %s --top vpi_test --max-time=1000 --vpi=%t.vpi.so 2>&1 | FileCheck %s

// CHECK: VPI_TEST: start_of_simulation callback fired
// CHECK: VPI_TEST: product=circt-sim
// CHECK: VPI_TEST: module name={{.*}}
// CHECK: VPI_TEST: found {{[1-9][0-9]*}} signals
// CHECK: VPI_TEST: time=0:0
// CHECK: VPI_TEST: {{[0-9]+}} passed, 0 failed
// CHECK: VPI_TEST: end_of_simulation callback fired
// CHECK: VPI_TEST: FINAL: {{[0-9]+}} passed, 0 failed

// Simple module with signals for VPI to discover.
hw.module @vpi_test(in %clk : i1, in %rst : i1) {
  %c0_i8 = hw.constant 0 : i8
  %counter = llhd.sig %c0_i8 : i8
}
