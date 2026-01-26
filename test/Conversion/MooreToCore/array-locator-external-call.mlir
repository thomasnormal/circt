// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Test array locator with external values that reference outer scope.
// This exercises the fix for SSA value invalidation when predicate contains
// references to values defined outside the predicate region.

// Test: Array locator with external constant reference
// The constant is defined outside the predicate body but used inside.
// When the predicate only uses constants, the optimized function call path is taken.
// CHECK-LABEL: hw.module @test_external_constant
// CHECK: llvm.call @__moore_array_find_eq
moore.module @test_external_constant() {
  %queue_var = moore.variable : <queue<i32, 0>>
  %queue = moore.read %queue_var : <queue<i32, 0>>
  %result_var = moore.variable : <queue<i32, 0>>

  // This constant is defined outside the predicate
  %external_const = moore.constant 42 : i32

  %result = moore.array.locator all, elements %queue : queue<i32, 0> -> <i32, 0> {
  ^bb0(%item: !moore.i32):
    // Reference the external constant inside the predicate
    %cond = moore.eq %item, %external_const : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  moore.blocking_assign %result_var, %result : queue<i32, 0>
  moore.output
}

// Test: Array locator with external variable read
// The variable is read outside but used in the predicate.
// When the predicate uses a variable read, the inline loop path is taken.
// CHECK-LABEL: hw.module @test_external_variable
// CHECK: llhd.prb %threshold_var
// CHECK: scf.for
// CHECK: comb.icmp eq
// CHECK: scf.if
moore.module @test_external_variable() {
  %queue_var = moore.variable : <queue<i32, 0>>
  %queue = moore.read %queue_var : <queue<i32, 0>>
  %threshold_var = moore.variable : <i32>
  %result_var = moore.variable : <queue<i32, 0>>

  // Read the threshold outside the predicate
  %threshold = moore.read %threshold_var : <i32>

  %result = moore.array.locator all, elements %queue : queue<i32, 0> -> <i32, 0> {
  ^bb0(%item: !moore.i32):
    // Reference the external value inside the predicate
    %cond = moore.eq %item, %threshold : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  moore.blocking_assign %result_var, %result : queue<i32, 0>
  moore.output
}
