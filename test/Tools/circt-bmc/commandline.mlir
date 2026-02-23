// RUN: circt-bmc --help | FileCheck %s

// CHECK: OVERVIEW: circt-bmc - bounded model checker
// CHECK-DAG: --allow-multi-clock
// CHECK-DAG: --rising-clocks-only
// CHECK-DAG: --print-counterexample
// CHECK-DAG: --assume-known-inputs
// CHECK-DAG: --x-optimistic
// CHECK-DAG: --fail-on-violation
// CHECK-DAG: --flatten-modules
// CHECK-DAG: --induction
// CHECK-DAG: --k-induction
// CHECK-DAG: --liveness
// CHECK-DAG: --liveness-lasso
// CHECK-DAG: --print-solver-output
// CHECK-DAG: --prune-bmc-registers
// CHECK-DAG: --emit-smtlib
