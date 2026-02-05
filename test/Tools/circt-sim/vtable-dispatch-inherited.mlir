// RUN: circt-sim %s | FileCheck %s

// Test vtable dispatch where a derived class inherits virtual methods
// without overriding them. This exercises the ClassNewOpConversion path
// that populates circt.vtable_entries for placeholder vtable globals
// when no moore.vtable op exists for the class.
//
// Object layout:
//   Derived: { Base: { i32 typeId, ptr vtablePtr }, i64 data }
//
// Both classes share the same virtual methods (speak, legs) but
// child_class uses the vtable populated from base_class method targets.

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  // Base class virtual methods
  func.func private @"base::speak"(%self: !llvm.ptr) {
    %msg = sim.fmt.literal "base speak\0A"
    sim.proc.print %msg
    return
  }

  func.func private @"base::legs"(%self: !llvm.ptr) -> i32 {
    %c2 = arith.constant 2 : i32
    return %c2 : i32
  }

  // Base vtable (has entries)
  llvm.mlir.global internal @"base::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [0, @"base::speak"],
      [1, @"base::legs"]
    ]
  } : !llvm.array<2 x ptr>

  // Derived vtable - populated from inherited methods (same targets as base)
  // This is the key test: the derived class doesn't override any methods,
  // so its vtable entries point to the base implementations.
  llvm.mlir.global internal @"derived::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [0, @"base::speak"],
      [1, @"base::legs"]
    ]
  } : !llvm.array<2 x ptr>

  // Global to hold instance
  llvm.mlir.global internal @"the_obj"(0 : i64) {addr_space = 0 : i32} : !llvm.ptr

  // Derived constructor
  func.func private @"derived::new"() -> !llvm.ptr {
    // struct Derived { struct Base { i32, ptr }, i64 }
    %sz = llvm.mlir.constant(20 : i64) : i64
    %obj = llvm.call @malloc(%sz) : (i64) -> !llvm.ptr

    // Store typeId
    %tid = arith.constant 2 : i32
    %tid_field = llvm.getelementptr %obj[0, 0, 0]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"derived", (struct<"base", (i32, ptr)>, i64)>
    llvm.store %tid, %tid_field : i32, !llvm.ptr

    // Store derived vtable pointer
    %vt = llvm.mlir.addressof @"derived::__vtable__" : !llvm.ptr
    %vt_field = llvm.getelementptr %obj[0, 0, 1]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"derived", (struct<"base", (i32, ptr)>, i64)>
    llvm.store %vt, %vt_field : !llvm.ptr, !llvm.ptr

    return %obj : !llvm.ptr
  }

  // LLVM global constructor
  llvm.func internal @"__ctor"() {
    %obj = func.call @"derived::new"() : () -> !llvm.ptr
    %global = llvm.mlir.addressof @"the_obj" : !llvm.ptr
    llvm.store %obj, %global : !llvm.ptr, !llvm.ptr
    llvm.return
  }

  llvm.mlir.global_ctors ctors = [@"__ctor"], priorities = [65535 : i32], data = [#llvm.zero]

  hw.module @inherited_vtable_test() {
    %fmt_prefix = sim.fmt.literal "legs = "
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      // Load from global (set by ctor)
      %global = llvm.mlir.addressof @"the_obj" : !llvm.ptr
      %obj = llvm.load %global : !llvm.ptr -> !llvm.ptr

      // Dispatch speak() through base vtable layout
      %vt_field = llvm.getelementptr %obj[0, 0, 1]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"derived", (struct<"base", (i32, ptr)>, i64)>
      %vptr = llvm.load %vt_field : !llvm.ptr -> !llvm.ptr

      %slot0 = llvm.getelementptr %vptr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fptr0 = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
      %speak = builtin.unrealized_conversion_cast %fptr0 : !llvm.ptr to (!llvm.ptr) -> ()
      func.call_indirect %speak(%obj) : (!llvm.ptr) -> ()

      // Dispatch legs() through base vtable layout
      %slot1 = llvm.getelementptr %vptr[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fptr1 = llvm.load %slot1 : !llvm.ptr -> !llvm.ptr
      %legs_fn = builtin.unrealized_conversion_cast %fptr1 : !llvm.ptr to (!llvm.ptr) -> i32
      %legs = func.call_indirect %legs_fn(%obj) : (!llvm.ptr) -> i32

      %fmt_val = sim.fmt.dec %legs specifierWidth 0 : i32
      %fmt_full = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
      sim.proc.print %fmt_full

      llhd.halt
    }
    hw.output
  }
}

// CHECK: base speak
// CHECK: legs = 2
