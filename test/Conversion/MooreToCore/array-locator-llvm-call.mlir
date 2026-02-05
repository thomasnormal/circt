// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Test array locator predicate containing llvm.call and unrealized casts.
// This mirrors process::status lowering inside UVM queue find predicates.

llvm.func @__moore_process_status(i64) -> i32

// CHECK-LABEL: hw.module @test_array_locator_llvm_call
// CHECK: llvm.call @__moore_process_status
// CHECK-NOT: moore.array.locator
moore.module @test_array_locator_llvm_call() {
  %queue_var = moore.variable : <queue<i32, 0>>
  %queue = moore.read %queue_var : <queue<i32, 0>>
  %result_var = moore.variable : <queue<i32, 0>>

  %result = moore.array.locator all, elements %queue : queue<i32, 0> -> <i32, 0> {
  ^bb0(%item: !moore.i32):
    %handle = moore.constant 0 : i32
    %handle_i64 = builtin.unrealized_conversion_cast %handle : !moore.i32 to i64
    %status_i32 = llvm.call @__moore_process_status(%handle_i64) : (i64) -> i32
    %status = builtin.unrealized_conversion_cast %status_i32 : i32 to !moore.i32
    %cond = moore.eq %item, %status : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  moore.blocking_assign %result_var, %result : queue<i32, 0>
  moore.output
}
