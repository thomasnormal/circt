// RUN: circt-bmc -b 1 --emit-smtlib --assume-known-inputs --module top %s | FileCheck %s

hw.module @top(in %sig: !hw.struct<value: i1, unknown: i1>, out out: i1) {
  %val = hw.struct_extract %sig["value"] : !hw.struct<value: i1, unknown: i1>
  verif.assert %val : i1
  hw.output %val : i1
}

// CHECK: (declare-const sig (_ BitVec 2))
// CHECK: ((_ extract 0 0) sig)
// CHECK: #b0
