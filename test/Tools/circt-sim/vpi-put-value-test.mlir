// RUN: cc -shared -fPIC -o %t.vpi.so %S/vpi-put-value-test.c -ldl
// RUN: circt-sim %s --top put_test --max-time=1000 --vpi=%t.vpi.so 2>&1 | FileCheck %s

// CHECK: VPI_PUT: start_of_simulation
// CHECK: VPI_PUT: testing signal={{.*}} width=8
// CHECK: VPI_PUT: initial_value=0
// CHECK: VPI_PUT: after_write=42
// CHECK: VPI_PUT: binary=
// CHECK: VPI_PUT: full_name=
// CHECK: VPI_PUT: found_value=42
// CHECK: VPI_PUT: {{[0-9]+}} passed, 0 failed
// CHECK: VPI_PUT: FINAL: {{[0-9]+}} passed, 0 failed

// Module with 8-bit port to test put_value/get_value roundtrip.
hw.module @put_test(in %clk : i1, in %data : i8) {
}
