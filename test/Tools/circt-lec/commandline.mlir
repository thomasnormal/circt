// RUN: circt-lec --help | FileCheck %s

// CHECK: OVERVIEW: circt-lec - logical equivalence checker
// CHECK-DAG: --accept-xprop-only
// CHECK-DAG: --accept-llhd-abstraction
// CHECK-DAG: --approx-temporal
// CHECK-DAG: --lec-approx
// CHECK-DAG: --lec-canonicalizer-max-iterations
// CHECK-DAG: --lec-canonicalizer-max-num-rewrites
// CHECK-DAG: --lec-strict
// CHECK-DAG: --strict-llhd
