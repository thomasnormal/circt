// RUN: circt-bmc -b 1 --emit-mlir --module top %s | FileCheck %s

hw.module @top() {
  %true = hw.constant true
  verif.cover %true : i1
  hw.output
}

// The JIT-style (non-SMTLIB-export) BMC lowering returns `true` when no
// "interesting" condition was found. Covers are interesting when hit, so the
// cover-hit condition must be inverted before yielding the solver result.
//
// CHECK: %true = arith.constant true
// CHECK: %[[HIT:.*]] = scf.for
// CHECK: smt.assert %[[HIT]]
// CHECK: %[[RES:.*]] = smt.check
// CHECK: %[[INV:.*]] = arith.xori %[[RES]], %true : i1
// CHECK: smt.yield %[[INV]] : i1
