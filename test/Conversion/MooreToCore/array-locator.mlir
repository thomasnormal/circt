// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Test array locator with external variable reference in predicate
// CHECK-LABEL: hw.module @test_array_locator_variable
moore.module @test_array_locator_variable() {
  // Variable to hold the comparison value
  %cmp_val = moore.variable : <i32>
  %val_5 = moore.constant 5 : i32
  moore.blocking_assign %cmp_val, %val_5 : i32

  // Queue input
  %queue_var = moore.variable : <queue<i32, 0>>

  // Read the queue
  %queue = moore.read %queue_var : <queue<i32, 0>>

  // Array locator with external variable reference (uses inline loop approach)
  // CHECK: scf.for
  // CHECK: llhd.prb
  // CHECK: comb.icmp eq
  // CHECK: scf.if
  %result = moore.array.locator all, elements %queue : queue<i32, 0> -> <i32, 0> {
  ^bb0(%item: !moore.i32):
    %cmp = moore.read %cmp_val : <i32>
    %cond = moore.eq %item, %cmp : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  // Store result
  %result_var = moore.variable : <queue<i32, 0>>
  moore.blocking_assign %result_var, %result : queue<i32, 0>
  moore.output
}

// Test array locator with constant comparison (uses runtime function)
// CHECK-LABEL: hw.module @test_array_locator_constant
moore.module @test_array_locator_constant() {
  // Queue input
  %queue_var = moore.variable : <queue<i32, 0>>
  %queue = moore.read %queue_var : <queue<i32, 0>>

  // Array locator with constant comparison
  // CHECK: llvm.call @__moore_array_find_eq
  %result = moore.array.locator all, elements %queue : queue<i32, 0> -> <i32, 0> {
  ^bb0(%item: !moore.i32):
    %five = moore.constant 5 : i32
    %cond = moore.eq %item, %five : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  %result_var = moore.variable : <queue<i32, 0>>
  moore.blocking_assign %result_var, %result : queue<i32, 0>
  moore.output
}

// Test array locator with "first" mode
// CHECK-LABEL: hw.module @test_array_locator_first
moore.module @test_array_locator_first() {
  %queue_var = moore.variable : <queue<i32, 0>>
  %queue = moore.read %queue_var : <queue<i32, 0>>
  %cmp_val = moore.variable : <i32>

  // Array locator with first mode and variable comparison
  // CHECK: scf.for
  %result = moore.array.locator first, elements %queue : queue<i32, 0> -> <i32, 0> {
  ^bb0(%item: !moore.i32):
    %cmp = moore.read %cmp_val : <i32>
    %cond = moore.eq %item, %cmp : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  %result_var = moore.variable : <queue<i32, 0>>
  moore.blocking_assign %result_var, %result : queue<i32, 0>
  moore.output
}

// Test array locator returning indices
// CHECK-LABEL: hw.module @test_array_locator_indices
moore.module @test_array_locator_indices() {
  %queue_var = moore.variable : <queue<i32, 0>>
  %queue = moore.read %queue_var : <queue<i32, 0>>
  %cmp_val = moore.variable : <i32>

  // Array locator returning indices
  // CHECK: scf.for
  %result = moore.array.locator all, indices %queue : queue<i32, 0> -> <i32, 0> {
  ^bb0(%item: !moore.i32):
    %cmp = moore.read %cmp_val : <i32>
    %cond = moore.eq %item, %cmp : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  %result_var = moore.variable : <queue<i32, 0>>
  moore.blocking_assign %result_var, %result : queue<i32, 0>
  moore.output
}

// Test array locator with packed struct element type and field access
// This tests the fix for handling packed structs (moore::StructType) in
// the inline conversion path of ArrayLocatorOpConversion.
// Previously, this would crash with "cast<UnpackedStructType> failed" because
// the code assumed only unpacked structs could appear in LLVM lowering.
// CHECK-LABEL: hw.module @test_array_locator_packed_struct
moore.module @test_array_locator_packed_struct() {
  // Queue of packed structs
  %queue_var = moore.variable : <queue<struct<{x: i32, y: i32}>, 0>>
  %queue = moore.read %queue_var : <queue<struct<{x: i32, y: i32}>, 0>>

  // Array locator with field access on packed struct
  // The predicate accesses item.x which requires proper handling of packed struct types
  // CHECK: scf.for
  // CHECK: llvm.load
  // CHECK: llvm.extractvalue
  %result = moore.array.locator first, elements %queue : queue<struct<{x: i32, y: i32}>, 0> -> <struct<{x: i32, y: i32}>, 0> {
  ^bb0(%item: !moore.struct<{x: i32, y: i32}>):
    %x_field = moore.struct_extract %item, "x" : struct<{x: i32, y: i32}> -> i32
    %one = moore.constant 1 : i32
    %cond = moore.eq %x_field, %one : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  %result_var = moore.variable : <queue<struct<{x: i32, y: i32}>, 0>>
  moore.blocking_assign %result_var, %result : queue<struct<{x: i32, y: i32}>, 0>
  moore.output
}
