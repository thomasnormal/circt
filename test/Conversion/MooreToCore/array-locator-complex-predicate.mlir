// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Declare class for testing complex predicates with class property access
moore.class.classdecl @QosEntry {
  moore.class.propertydecl @awid : !moore.i32
  moore.class.propertydecl @priority : !moore.i32
}

// Test array locator with simpler complex predicate: item.field == constant
// Uses optimized runtime path for simple field comparisons.
// CHECK-LABEL: hw.module @test_field_eq_constant
// CHECK: llvm.call @__moore_array_find_field_cmp
moore.module @test_field_eq_constant() {
  %qos_queue_var = moore.variable : <queue<class<@QosEntry>, 0>>
  %qos_queue = moore.read %qos_queue_var : <queue<class<@QosEntry>, 0>>
  %result_var = moore.variable : <queue<class<@QosEntry>, 0>>

  %result = moore.array.locator all, elements %qos_queue : queue<class<@QosEntry>, 0> -> <class<@QosEntry>, 0> {
  ^bb0(%item: !moore.class<@QosEntry>):
    %awid_ref = moore.class.property_ref %item[@awid] : <@QosEntry> -> !moore.ref<i32>
    %awid = moore.read %awid_ref : <i32>
    %target = moore.constant 42 : i32
    %cond = moore.eq %awid, %target : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  moore.blocking_assign %result_var, %result : queue<class<@QosEntry>, 0>
  moore.output
}

// Test array locator with logical AND in predicate - triggers inline conversion
// because it's not a simple field comparison pattern.
// CHECK-LABEL: hw.module @test_predicate_and
// CHECK: scf.for
// CHECK: llvm.getelementptr
// CHECK: llvm.load
// CHECK: comb.icmp eq
// CHECK: comb.icmp sgt
// CHECK: comb.and
// CHECK: scf.if
moore.module @test_predicate_and() {
  %qos_queue_var = moore.variable : <queue<class<@QosEntry>, 0>>
  %qos_queue = moore.read %qos_queue_var : <queue<class<@QosEntry>, 0>>
  %result_var = moore.variable : <queue<class<@QosEntry>, 0>>

  %result = moore.array.locator all, elements %qos_queue : queue<class<@QosEntry>, 0> -> <class<@QosEntry>, 0> {
  ^bb0(%item: !moore.class<@QosEntry>):
    %awid_ref = moore.class.property_ref %item[@awid] : <@QosEntry> -> !moore.ref<i32>
    %awid = moore.read %awid_ref : <i32>
    %priority_ref = moore.class.property_ref %item[@priority] : <@QosEntry> -> !moore.ref<i32>
    %priority = moore.read %priority_ref : <i32>

    %c10 = moore.constant 10 : i32
    %c5 = moore.constant 5 : i32
    %cond1 = moore.eq %awid, %c10 : i32 -> i1
    %cond2 = moore.sgt %priority, %c5 : i32 -> i1
    %cond = moore.and %cond1, %cond2 : i1
    moore.array.locator.yield %cond : i1
  }

  moore.blocking_assign %result_var, %result : queue<class<@QosEntry>, 0>
  moore.output
}

// Test array locator with relational operators - uses optimized runtime path
// CHECK-LABEL: hw.module @test_relational_ops
// CHECK: llvm.call @__moore_array_find_cmp
moore.module @test_relational_ops() {
  %queue_var = moore.variable : <queue<i32, 0>>
  %queue = moore.read %queue_var : <queue<i32, 0>>
  %result_var = moore.variable : <queue<i32, 0>>

  %result = moore.array.locator all, elements %queue : queue<i32, 0> -> <i32, 0> {
  ^bb0(%item: !moore.i32):
    %threshold = moore.constant 100 : i32
    %cond = moore.sgt %item, %threshold : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  moore.blocking_assign %result_var, %result : queue<i32, 0>
  moore.output
}

// Test array locator with ne (not equal) operator - uses optimized runtime path
// CHECK-LABEL: hw.module @test_ne_operator
// CHECK: llvm.call @__moore_array_find_cmp
moore.module @test_ne_operator() {
  %queue_var = moore.variable : <queue<i32, 0>>
  %queue = moore.read %queue_var : <queue<i32, 0>>
  %result_var = moore.variable : <queue<i32, 0>>

  %result = moore.array.locator all, elements %queue : queue<i32, 0> -> <i32, 0> {
  ^bb0(%item: !moore.i32):
    %target = moore.constant 0 : i32
    %cond = moore.ne %item, %target : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  moore.blocking_assign %result_var, %result : queue<i32, 0>
  moore.output
}

// Test array locator with addition in predicate - triggers inline conversion
// CHECK-LABEL: hw.module @test_add_in_predicate
// CHECK: scf.for
// CHECK: comb.add
// CHECK: comb.icmp eq
// CHECK: scf.if
moore.module @test_add_in_predicate() {
  %queue_var = moore.variable : <queue<i32, 0>>
  %queue = moore.read %queue_var : <queue<i32, 0>>
  %result_var = moore.variable : <queue<i32, 0>>

  %result = moore.array.locator all, elements %queue : queue<i32, 0> -> <i32, 0> {
  ^bb0(%item: !moore.i32):
    %one = moore.constant 1 : i32
    %item_plus_1 = moore.add %item, %one : i32
    %target = moore.constant 100 : i32
    %cond = moore.eq %item_plus_1, %target : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  moore.blocking_assign %result_var, %result : queue<i32, 0>
  moore.output
}
