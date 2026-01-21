// RUN: circt-opt %s --coverage-instrument | FileCheck %s

// CHECK-LABEL: hw.module @TestModule
hw.module @TestModule(in %x: i4, in %y: i4, out u: i4) {
  // CHECK: coverage.toggle %x name "x" hierarchy "TestModule"
  // CHECK: coverage.toggle %y name "y" hierarchy "TestModule"
  // CHECK: coverage.line
  %0 = comb.add %x, %y : i4
  // CHECK: coverage.toggle %{{.*}} name "u" hierarchy "TestModule"
  // CHECK: hw.output
  hw.output %0 : i4
}
// CHECK: }

// Test with toggle coverage disabled
// RUN: circt-opt %s --coverage-instrument=toggle=false | FileCheck %s --check-prefix=NO-TOGGLE

// NO-TOGGLE-LABEL: hw.module @TestModule
// NO-TOGGLE: coverage.line
// NO-TOGGLE-NOT: coverage.toggle
// NO-TOGGLE: hw.output
