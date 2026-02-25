// RUN: circt-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: hw.module @extractConcatSelfCycle
hw.module @extractConcatSelfCycle(in %in : i1, out out : i1) {
  // Keep this cycle intact; canonicalization must not crash by trying to
  // replace %x with itself.
  // CHECK: %[[CAT:.*]] = comb.concat %[[X:.*]], %in : i1, i1
  // CHECK-NEXT: %[[X]] = comb.extract %[[CAT]] from 1 : (i2) -> i1
  %cat = comb.concat %x, %in : i1, i1
  %x = comb.extract %cat from 1 : (i2) -> i1
  hw.output %x : i1
}
