// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @assert_enable
// CHECK: smt.or
// CHECK: smt.assert
func.func @assert_enable(%arg0: i1, %en: i1) {
  verif.assert %arg0 if %en : i1
  return
}

// -----

// CHECK-LABEL: func.func @assume_enable
// CHECK: smt.or
// CHECK: smt.assert
func.func @assume_enable(%arg0: i1, %en: i1) {
  verif.assume %arg0 if %en : i1
  return
}

// -----

// CHECK-LABEL: func.func @cover_enable
// CHECK: smt.and
// CHECK: smt.assert
func.func @cover_enable(%arg0: i1, %en: i1) {
  verif.cover %arg0 if %en : i1
  return
}
