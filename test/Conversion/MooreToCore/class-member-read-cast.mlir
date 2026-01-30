// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that class member reads use llvm.load through unrealized_conversion_cast
// This validates the ReadOpConversion fix that looks through unrealized casts
// to find LLVM pointers from class member GEP operations.

//===----------------------------------------------------------------------===//
// Class declaration for testing
//===----------------------------------------------------------------------===//

moore.class.classdecl @TestClass {
  moore.class.propertydecl @intField : !moore.i32
  moore.class.propertydecl @logicField : !moore.l64
}

//===----------------------------------------------------------------------===//
// Test: Read from class member should use llvm.load, not llhd.prb
//===----------------------------------------------------------------------===//

// The key fix is that ReadOpConversion checks for unrealized_conversion_cast
// wrapping an LLVM pointer (from GEP), and uses llvm.load instead of llhd.prb.
// Previously, this would incorrectly use llhd.prb which doesn't work for class
// memory that isn't backed by LLHD signals.

// CHECK-LABEL: func.func @test_class_member_read
// CHECK-SAME: (%[[THIS:arg0]]: !llvm.ptr) -> i32
// CHECK:   %[[GEP:.*]] = llvm.getelementptr %[[THIS]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"TestClass"
// CHECK:   %[[LOAD:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> i32
// CHECK:   return %[[LOAD]] : i32
// CHECK-NOT: llhd.prb
func.func @test_class_member_read(%this: !moore.class<@TestClass>) -> !moore.i32 {
  %ref = moore.class.property_ref %this[@intField] : <@TestClass> -> !moore.ref<i32>
  %val = moore.read %ref : !moore.ref<i32>
  return %val : !moore.i32
}

//===----------------------------------------------------------------------===//
// Test: Read from 4-state class member uses llvm.load with cast back to HW type
//===----------------------------------------------------------------------===//

// For 4-state types (like l64), the loaded LLVM struct needs to be cast back
// to the HW struct type.

// CHECK-LABEL: func.func @test_class_member_read_fourstate
// CHECK-SAME: (%[[THIS:arg0]]: !llvm.ptr)
// CHECK:   %[[GEP:.*]] = llvm.getelementptr %[[THIS]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"TestClass"
// CHECK:   llvm.load %[[GEP]]
// CHECK-NOT: llhd.prb
func.func @test_class_member_read_fourstate(%this: !moore.class<@TestClass>) -> !moore.l64 {
  %ref = moore.class.property_ref %this[@logicField] : <@TestClass> -> !moore.ref<l64>
  %val = moore.read %ref : !moore.ref<l64>
  return %val : !moore.l64
}

//===----------------------------------------------------------------------===//
// Test: Multiple reads from same class instance
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_multiple_reads
// CHECK-SAME: (%[[THIS:arg0]]: !llvm.ptr)
// CHECK:   llvm.getelementptr %[[THIS]][0, 2]
// CHECK:   llvm.load
// CHECK:   llvm.getelementptr %[[THIS]][0, 3]
// CHECK:   llvm.load
// CHECK-NOT: llhd.prb
func.func @test_multiple_reads(%this: !moore.class<@TestClass>) -> (!moore.i32, !moore.l64) {
  %ref1 = moore.class.property_ref %this[@intField] : <@TestClass> -> !moore.ref<i32>
  %val1 = moore.read %ref1 : !moore.ref<i32>
  %ref2 = moore.class.property_ref %this[@logicField] : <@TestClass> -> !moore.ref<l64>
  %val2 = moore.read %ref2 : !moore.ref<l64>
  return %val1, %val2 : !moore.i32, !moore.l64
}
