// RUN: circt-sim %s | FileCheck %s

// Test full vtable dispatch through a heap-allocated object.
// Exercises the complete chain:
//   1. Allocate object with llvm.call @malloc
//   2. Store vtable pointer into object via llvm.addressof + llvm.store
//   3. Load vtable pointer back from object via llvm.load
//   4. Index into vtable array with llvm.getelementptr
//   5. Load function pointer from vtable slot via llvm.load
//   6. Call through function pointer with func.call_indirect
//
// Object layout: !llvm.struct<"animal", (ptr)>
//   field 0: vtable pointer (!llvm.ptr -> !llvm.array<2 x ptr>)
//
// Vtable layout: !llvm.array<2 x ptr>
//   slot 0: @"animal::speak"
//   slot 1: @"animal::legs"

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  // Virtual method: speak - prints a string
  func.func private @"animal::speak"(%self: !llvm.ptr) {
    %msg = sim.fmt.literal "woof\0A"
    sim.proc.print %msg
    return
  }

  // Virtual method: legs - returns an integer
  func.func private @"animal::legs"(%self: !llvm.ptr) -> i32 {
    %c4 = arith.constant 4 : i32
    return %c4 : i32
  }

  // Vtable global initialized via circt.vtable_entries
  llvm.mlir.global internal @"animal::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [0, @"animal::speak"],
      [1, @"animal::legs"]
    ]
  } : !llvm.array<2 x ptr>

  hw.module @vtable_dispatch_test() {
    %fmt_legs_prefix = sim.fmt.literal "legs = "
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      // --- Allocate object ---
      %obj_size = llvm.mlir.constant(8 : i64) : i64
      %obj = llvm.call @malloc(%obj_size) : (i64) -> !llvm.ptr

      // --- Store vtable pointer into object field 0 ---
      %vtable_addr = llvm.mlir.addressof @"animal::__vtable__" : !llvm.ptr
      %vptr_field = llvm.getelementptr %obj[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"animal", (ptr)>
      llvm.store %vtable_addr, %vptr_field : !llvm.ptr, !llvm.ptr

      // --- Dispatch virtual method: speak (slot 0) ---
      // Load vtable pointer from object
      %vptr0 = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr
      // Index into vtable at slot 0
      %slot0_addr = llvm.getelementptr %vptr0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      // Load function pointer
      %fptr0 = llvm.load %slot0_addr : !llvm.ptr -> !llvm.ptr
      // Cast and call
      %speak = builtin.unrealized_conversion_cast %fptr0 : !llvm.ptr to (!llvm.ptr) -> ()
      func.call_indirect %speak(%obj) : (!llvm.ptr) -> ()

      // --- Dispatch virtual method: legs (slot 1) ---
      %vptr1 = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr
      %slot1_addr = llvm.getelementptr %vptr1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fptr1 = llvm.load %slot1_addr : !llvm.ptr -> !llvm.ptr
      %legs_fn = builtin.unrealized_conversion_cast %fptr1 : !llvm.ptr to (!llvm.ptr) -> i32
      %leg_count = func.call_indirect %legs_fn(%obj) : (!llvm.ptr) -> i32

      // Print the result
      %fmt_val = sim.fmt.dec %leg_count specifierWidth 0 : i32
      %fmt_full = sim.fmt.concat (%fmt_legs_prefix, %fmt_val, %fmt_nl)
      sim.proc.print %fmt_full

      llhd.halt
    }
    hw.output
  }
}

// CHECK: woof
// CHECK: legs = 4
