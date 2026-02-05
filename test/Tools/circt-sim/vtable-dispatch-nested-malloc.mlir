// RUN: circt-sim %s | FileCheck %s

// Test vtable dispatch where object pointer is stored inside another
// malloc'd object (double indirection). This mimics UVM patterns where
// class members hold references to other class objects.
//
// Container layout: !llvm.struct<"container", (ptr)>
//   field 0: pointer to inner object
//
// Inner layout: !llvm.struct<"inner", (i32, ptr)>
//   field 0: type_id (i32)
//   field 1: vtable pointer

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  func.func private @"inner::get_value"(%self: !llvm.ptr) -> i32 {
    %c77 = arith.constant 77 : i32
    return %c77 : i32
  }

  llvm.mlir.global internal @"inner::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [0, @"inner::get_value"]
    ]
  } : !llvm.array<1 x ptr>

  // Create inner object
  func.func private @"inner::new"() -> !llvm.ptr {
    %sz = llvm.mlir.constant(16 : i64) : i64
    %obj = llvm.call @malloc(%sz) : (i64) -> !llvm.ptr
    %tid = arith.constant 5 : i32
    %f0 = llvm.getelementptr %obj[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"inner", (i32, ptr)>
    llvm.store %tid, %f0 : i32, !llvm.ptr
    %vt = llvm.mlir.addressof @"inner::__vtable__" : !llvm.ptr
    %f1 = llvm.getelementptr %obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"inner", (i32, ptr)>
    llvm.store %vt, %f1 : !llvm.ptr, !llvm.ptr
    return %obj : !llvm.ptr
  }

  // Create container and store inner object in it
  func.func private @"container::new"() -> !llvm.ptr {
    %sz = llvm.mlir.constant(8 : i64) : i64
    %container = llvm.call @malloc(%sz) : (i64) -> !llvm.ptr
    %inner = func.call @"inner::new"() : () -> !llvm.ptr
    %f0 = llvm.getelementptr %container[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"container", (ptr)>
    llvm.store %inner, %f0 : !llvm.ptr, !llvm.ptr
    return %container : !llvm.ptr
  }

  // Function that loads inner from container and calls virtual method
  func.func private @"use_container"(%container: !llvm.ptr) {
    %fmt_prefix = sim.fmt.literal "nested value = "
    %fmt_nl = sim.fmt.literal "\0A"

    // Load inner object pointer from container
    %f0 = llvm.getelementptr %container[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"container", (ptr)>
    %inner = llvm.load %f0 : !llvm.ptr -> !llvm.ptr

    // Virtual dispatch: load vtable pointer from inner
    %vf = llvm.getelementptr %inner[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"inner", (i32, ptr)>
    %vptr = llvm.load %vf : !llvm.ptr -> !llvm.ptr

    // Index into vtable slot 0
    %slot = llvm.getelementptr %vptr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot : !llvm.ptr -> !llvm.ptr

    // Cast and call
    %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr) -> i32
    %val = func.call_indirect %fn(%inner) : (!llvm.ptr) -> i32

    %fmt_val = sim.fmt.dec %val specifierWidth 0 : i32
    %fmt_full = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_full
    return
  }

  hw.module @vtable_nested_test() {
    llhd.process {
      %container = func.call @"container::new"() : () -> !llvm.ptr
      func.call @"use_container"(%container) : (!llvm.ptr) -> ()
      llhd.halt
    }
    hw.output
  }
}

// CHECK: nested value = 77
