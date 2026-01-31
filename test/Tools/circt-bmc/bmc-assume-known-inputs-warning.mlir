// RUN: circt-bmc -b 1 --emit-mlir --assume-known-inputs --module top %s 2>&1 | FileCheck %s
// RUN: circt-bmc -b 1 --emit-mlir --module top %s 2>&1 | FileCheck %s --check-prefix=WARN

hw.module @top(in %sig: !hw.struct<value: i1, unknown: i1>, out out: i1) {
  %val = hw.struct_extract %sig["value"] : !hw.struct<value: i1, unknown: i1>
  verif.assert %val : i1
  hw.output %val : i1
}

// CHECK-NOT: 4-state inputs are unconstrained
// WARN: 4-state inputs are unconstrained
