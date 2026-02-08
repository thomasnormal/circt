// RUN: circt-sim %s 2>&1 | FileCheck %s

// Test that call_indirect absorbs internal failures from dispatched functions.
// When a function dispatched via vtable call_indirect fails internally (e.g.,
// due to an unresolvable operation), the failure should be absorbed with a
// warning instead of cascading up and halting the entire process.
//
// This pattern occurs in UVM phase traversal where recursive calls to
// traverse_on -> traverse -> traverse_on can cascade failures if any single
// component's phase method fails.
//
// The test sets up:
//   1. A "good" virtual method that prints a message
//   2. A "bad" virtual method that calls a function which fails
//   3. A caller that dispatches both methods - the bad one should warn but
//      execution should continue to the good one

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  // A function that always succeeds
  func.func private @"component::good_method"(%self: !llvm.ptr) {
    %msg = sim.fmt.literal "good_method called\0A"
    sim.proc.print %msg
    return
  }

  // A function that calls another function which will fail internally.
  // The inner function calls a function that doesn't exist via call_indirect
  // on a null/X function pointer, triggering the error path.
  func.func private @"component::bad_method"(%self: !llvm.ptr) {
    // Load a function pointer from a zeroed vtable slot (not registered in
    // addressToFunction), which will fail resolution.  Because this is inside
    // a func.call_indirect dispatch, the outer call_indirect should absorb
    // the failure.
    %vtable_addr = llvm.mlir.addressof @"empty_vtable" : !llvm.ptr
    %slot_addr = llvm.getelementptr %vtable_addr[0, 0]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot_addr : !llvm.ptr -> !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fptr
      : !llvm.ptr to (!llvm.ptr) -> ()
    // This call_indirect will either resolve to X or to address 0,
    // which isn't in addressToFunction.  The handler returns success()
    // with a warning.
    func.call_indirect %fn(%self) : (!llvm.ptr) -> ()
    // If the inner call_indirect is absorbed, we reach here
    %msg = sim.fmt.literal "bad_method survived inner failure\0A"
    sim.proc.print %msg
    return
  }

  // Vtable for "component" with both methods
  llvm.mlir.global internal @"component::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [0, @"component::bad_method"],
      [1, @"component::good_method"]
    ]
  } : !llvm.array<2 x ptr>

  // Empty vtable with no entries (all zeros) - used by bad_method to
  // create a function pointer that can't be resolved
  llvm.mlir.global internal @"empty_vtable"(#llvm.zero) {
    addr_space = 0 : i32
  } : !llvm.array<1 x ptr>

  hw.module @vtable_dispatch_internal_failure_test() {
    llhd.process {
      // Allocate object
      %obj_size = llvm.mlir.constant(8 : i64) : i64
      %obj = llvm.call @malloc(%obj_size) : (i64) -> !llvm.ptr

      // Store vtable pointer into object field 0
      %vtable_addr = llvm.mlir.addressof @"component::__vtable__" : !llvm.ptr
      %vptr_field = llvm.getelementptr %obj[0, 0]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"component", (ptr)>
      llvm.store %vtable_addr, %vptr_field : !llvm.ptr, !llvm.ptr

      // Dispatch bad_method (slot 0) - should warn but not halt
      %vptr0 = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr
      %slot0 = llvm.getelementptr %vptr0[0, 0]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fptr0 = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
      %bad_fn = builtin.unrealized_conversion_cast %fptr0
        : !llvm.ptr to (!llvm.ptr) -> ()
      func.call_indirect %bad_fn(%obj) : (!llvm.ptr) -> ()

      // Dispatch good_method (slot 1) - should execute normally
      %vptr1 = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr
      %slot1 = llvm.getelementptr %vptr1[0, 1]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fptr1 = llvm.load %slot1 : !llvm.ptr -> !llvm.ptr
      %good_fn = builtin.unrealized_conversion_cast %fptr1
        : !llvm.ptr to (!llvm.ptr) -> ()
      func.call_indirect %good_fn(%obj) : (!llvm.ptr) -> ()

      %done = sim.fmt.literal "dispatch sequence complete\0A"
      sim.proc.print %done

      llhd.halt
    }
    hw.output
  }
}

// The bad_method's inner call_indirect loads a zero function pointer from
// empty_vtable.  The handler detects address 0 / X and absorbs it.
// bad_method then prints its survival message.
// After bad_method returns, good_method is dispatched and prints its message.

// CHECK: bad_method survived inner failure
// CHECK: good_method called
// CHECK: dispatch sequence complete
