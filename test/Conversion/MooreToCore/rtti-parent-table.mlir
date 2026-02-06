// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Test that MooreToCore emits a circt.rtti_parent_table module attribute
// that maps typeId -> parentTypeId (0 = root).
//
// Class hierarchy:
//   A (root)           -> typeId=1, parentTypeId=0
//   B extends A        -> typeId=2, parentTypeId=1
//   C extends B        -> typeId=3, parentTypeId=2
//
// Expected table (indexed by typeId):
//   table[0] = 0  (sentinel/unused)
//   table[1] = 0  (A is root)
//   table[2] = 1  (B's parent is A with typeId 1)
//   table[3] = 2  (C's parent is B with typeId 2)

// CHECK: module attributes {circt.rtti_parent_table = dense<[0, 0, 1, 2]> : tensor<4xi32>}

moore.class.classdecl @A {
  moore.class.propertydecl @x : !moore.i32
}

moore.class.classdecl @B extends @A {
  moore.class.propertydecl @y : !moore.i32
}

moore.class.classdecl @C extends @B {
  moore.class.propertydecl @z : !moore.i32
}

// Trigger resolution of the entire hierarchy by allocating the leaf class.
func.func @test_rtti() -> !moore.class<@C> {
  %obj = moore.class.new : <@C>
  return %obj : !moore.class<@C>
}
