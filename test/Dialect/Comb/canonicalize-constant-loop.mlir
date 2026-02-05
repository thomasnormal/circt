// RUN: circt-opt %s --mlir-disable-threading --canonicalize='max-num-rewrites=2000 max-iterations=1 top-down=0 region-simplify=disabled' | FileCheck %s

// This is a regression test for a canonicalization non-convergence bug where
// `comb.or`/`comb.and` would rebuild the op with a freshly materialized
// identical constant operand on each rewrite, causing unbounded numbers of
// duplicate `hw.constant` ops.

module {
  // CHECK-LABEL: hw.module @or_const_loop
  hw.module @or_const_loop(in %a : i4, out out : i4) {
    // CHECK-COUNT-1: hw.constant -8 : i4
    %c-8_i4 = hw.constant -8 : i4
    %0 = comb.or %a, %c-8_i4 : i4
    hw.output %0 : i4
  }
}

