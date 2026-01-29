// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Test array locator on associative arrays with integer keys
// CHECK-LABEL: hw.module @test_assoc_array_find_first_index
// CHECK: llvm.call @__moore_assoc_size
// CHECK: scf.for
// CHECK: llvm.call @__moore_assoc_first
// CHECK: llvm.call @__moore_assoc_next
// CHECK: llvm.call @__moore_assoc_get_ref
moore.module @test_assoc_array_find_first_index() {
  // Associative array with integer keys
  %assoc_var = moore.variable : <assoc_array<i32, i32>>
  %assoc = moore.read %assoc_var : <assoc_array<i32, i32>>

  // Variable for comparison
  %target_var = moore.variable : <i32>
  %target = moore.read %target_var : <i32>

  // find_first_index on associative array
  %result = moore.array.locator first, indices %assoc : assoc_array<i32, i32> -> <i32, 0> {
  ^bb0(%item: !moore.i32, %index: !moore.i32):
    %cond = moore.eq %item, %target : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  %result_var = moore.variable : <queue<i32, 0>>
  moore.blocking_assign %result_var, %result : queue<i32, 0>
  moore.output
}

// Test array locator with find_first (returns elements)
// CHECK-LABEL: hw.module @test_assoc_array_find_first
// CHECK: scf.for
// CHECK: llvm.call @__moore_assoc_get_ref
moore.module @test_assoc_array_find_first() {
  %assoc_var = moore.variable : <assoc_array<i32, i32>>
  %assoc = moore.read %assoc_var : <assoc_array<i32, i32>>
  %target_var = moore.variable : <i32>
  %target = moore.read %target_var : <i32>

  // find_first on associative array - returns elements
  %result = moore.array.locator first, elements %assoc : assoc_array<i32, i32> -> <i32, 0> {
  ^bb0(%item: !moore.i32):
    %cond = moore.eq %item, %target : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  %result_var = moore.variable : <queue<i32, 0>>
  moore.blocking_assign %result_var, %result : queue<i32, 0>
  moore.output
}

// Test array locator with find (returns all matching)
// CHECK-LABEL: hw.module @test_assoc_array_find_all
// CHECK: scf.for
moore.module @test_assoc_array_find_all() {
  %assoc_var = moore.variable : <assoc_array<i32, i32>>
  %assoc = moore.read %assoc_var : <assoc_array<i32, i32>>
  %target_var = moore.variable : <i32>
  %target = moore.read %target_var : <i32>

  // find on associative array - returns all matching elements
  %result = moore.array.locator all, elements %assoc : assoc_array<i32, i32> -> <i32, 0> {
  ^bb0(%item: !moore.i32):
    %cond = moore.eq %item, %target : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  %result_var = moore.variable : <queue<i32, 0>>
  moore.blocking_assign %result_var, %result : queue<i32, 0>
  moore.output
}

// Test array locator with find_last_index
// CHECK-LABEL: hw.module @test_assoc_array_find_last_index
// CHECK: scf.for
moore.module @test_assoc_array_find_last_index() {
  %assoc_var = moore.variable : <assoc_array<i32, i32>>
  %assoc = moore.read %assoc_var : <assoc_array<i32, i32>>
  %target_var = moore.variable : <i32>
  %target = moore.read %target_var : <i32>

  // find_last_index - collects all matching indices and returns the last one
  %result = moore.array.locator last, indices %assoc : assoc_array<i32, i32> -> <i32, 0> {
  ^bb0(%item: !moore.i32, %index: !moore.i32):
    %cond = moore.eq %item, %target : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  %result_var = moore.variable : <queue<i32, 0>>
  moore.blocking_assign %result_var, %result : queue<i32, 0>
  moore.output
}
