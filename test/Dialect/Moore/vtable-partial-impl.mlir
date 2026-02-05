// RUN: circt-opt %s --moore-create-vtables --verify-diagnostics | FileCheck %s

// Test that classes with partial virtual method implementations still get
// vtables. Previously, the CreateVTables pass skipped any class that had
// *any* unimplemented inherited method (using allHaveImpl), which caused
// UVM classes like uvm_phase_hopper to have empty vtables.
// The fix changes the check to noneHaveImpl so that classes with at least
// some implemented methods still get vtables; unimplemented slots are
// simply omitted from the vtable.

// Base interface with two pure virtual methods (no impl)
moore.class.classdecl @BaseInterface {
  moore.class.methoddecl @method_a : (!moore.class<@BaseInterface>) -> ()
  moore.class.methoddecl @method_b : (!moore.class<@BaseInterface>) -> ()
}

// Derived class that implements only method_a but not method_b.
// This is the key scenario: PartialImpl has at least some implemented methods,
// so it should still participate in vtable generation (not be skipped).
moore.class.classdecl @PartialImpl implements [@BaseInterface] {
  moore.class.methoddecl @method_a -> @"PartialImpl::method_a" : (!moore.class<@PartialImpl>) -> ()
}
func.func private @"PartialImpl::method_a"(%arg0: !moore.class<@PartialImpl>) {
  return
}

// Fully abstract class (all pure virtual) should NOT get a vtable
// CHECK-NOT: moore.vtable @PureAbstract
moore.class.classdecl @PureAbstract {
  moore.class.methoddecl @abstract_method : (!moore.class<@PureAbstract>) -> ()
}

// Class extending PartialImpl that implements both methods.
// The vtable should contain entries for both methods from FullImpl's
// perspective, with the nested PartialImpl vtable only containing method_a
// (since PartialImpl only declares method_a).
// CHECK-LABEL: moore.vtable @FullImpl::@vtable {
// CHECK-NEXT:    moore.vtable @PartialImpl::@vtable {
// CHECK-NEXT:      moore.vtable @BaseInterface::@vtable {
// CHECK-NEXT:        moore.vtable_entry @method_a -> @"FullImpl::method_a"
// CHECK-NEXT:        moore.vtable_entry @method_b -> @"FullImpl::method_b"
// CHECK-NEXT:      }
// CHECK-NEXT:      moore.vtable_entry @method_a -> @"FullImpl::method_a"
// CHECK-NEXT:    }
// CHECK-NEXT:    moore.vtable_entry @method_a -> @"FullImpl::method_a"
// CHECK-NEXT:    moore.vtable_entry @method_b -> @"FullImpl::method_b"
// CHECK-NEXT:  }
moore.class.classdecl @FullImpl extends @PartialImpl {
  moore.class.methoddecl @method_a -> @"FullImpl::method_a" : (!moore.class<@FullImpl>) -> ()
  moore.class.methoddecl @method_b -> @"FullImpl::method_b" : (!moore.class<@FullImpl>) -> ()
}
func.func private @"FullImpl::method_a"(%arg0: !moore.class<@FullImpl>) {
  return
}
func.func private @"FullImpl::method_b"(%arg0: !moore.class<@FullImpl>) {
  return
}
