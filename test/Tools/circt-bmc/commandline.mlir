// RUN: circt-bmc --help | FileCheck %s

// CHECK: OVERVIEW: circt-bmc - bounded model checker
// CHECK-DAG: --allow-multi-clock
// CHECK-DAG: --rising-clocks-only
// CHECK-DAG: --print-counterexample
// CHECK-DAG: --assume-known-inputs
// CHECK-DAG: --fail-on-violation
// CHECK-DAG: --z3-path
// CHECK-DAG: run-smtlib
