// RUN: circt-sim %s | FileCheck %s

// Test vtable dispatch where the object is created in one function
// and the virtual method is called in another (like UVM patterns).
// This tests that malloc'd memory persists across function calls.

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  // Virtual method: get_name
  func.func private @"base::get_name"(%self: !llvm.ptr) -> !llvm.struct<(ptr, i64)> {
    %msg = sim.fmt.literal "base_obj\0A"
    sim.proc.print %msg
    %null = llvm.mlir.zero : !llvm.ptr
    %zero = llvm.mlir.constant(0 : i64) : i64
    %result = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %r1 = llvm.insertvalue %null, %result[0] : !llvm.struct<(ptr, i64)>
    %r2 = llvm.insertvalue %zero, %r1[1] : !llvm.struct<(ptr, i64)>
    return %r2 : !llvm.struct<(ptr, i64)>
  }

  // Virtual method: get_id
  func.func private @"base::get_id"(%self: !llvm.ptr) -> i32 {
    %c42 = arith.constant 42 : i32
    return %c42 : i32
  }

  // Vtable global
  llvm.mlir.global internal @"base::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [0, @"base::get_name"],
      [1, @"base::get_id"]
    ]
  } : !llvm.array<2 x ptr>

  // Constructor: creates object and stores vtable pointer
  // Object layout: !llvm.struct<"base", (i32, ptr)>
  //   field 0: type_id (i32)
  //   field 1: vtable pointer (!llvm.ptr)
  func.func private @"base::new"() -> !llvm.ptr {
    %obj_size = llvm.mlir.constant(16 : i64) : i64
    %obj = llvm.call @malloc(%obj_size) : (i64) -> !llvm.ptr

    // Store type ID
    %type_id = arith.constant 1 : i32
    %tid_field = llvm.getelementptr %obj[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"base", (i32, ptr)>
    llvm.store %type_id, %tid_field : i32, !llvm.ptr

    // Store vtable pointer
    %vtable_addr = llvm.mlir.addressof @"base::__vtable__" : !llvm.ptr
    %vptr_field = llvm.getelementptr %obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"base", (i32, ptr)>
    llvm.store %vtable_addr, %vptr_field : !llvm.ptr, !llvm.ptr

    return %obj : !llvm.ptr
  }

  // Function that takes an object and calls a virtual method on it
  func.func private @"call_get_name"(%obj: !llvm.ptr) {
    // Load vtable pointer from object field 1
    %vptr_field = llvm.getelementptr %obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"base", (i32, ptr)>
    %vptr = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr
    // Index into vtable at slot 0 (get_name)
    %slot_addr = llvm.getelementptr %vptr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
    %fptr = llvm.load %slot_addr : !llvm.ptr -> !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr) -> !llvm.struct<(ptr, i64)>
    %result = func.call_indirect %fn(%obj) : (!llvm.ptr) -> !llvm.struct<(ptr, i64)>
    return
  }

  // Function that calls get_id virtual method
  func.func private @"call_get_id"(%obj: !llvm.ptr) {
    %fmt_prefix = sim.fmt.literal "id = "
    %fmt_nl = sim.fmt.literal "\0A"

    %vptr_field = llvm.getelementptr %obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"base", (i32, ptr)>
    %vptr = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr
    %slot_addr = llvm.getelementptr %vptr[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
    %fptr = llvm.load %slot_addr : !llvm.ptr -> !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr) -> i32
    %id = func.call_indirect %fn(%obj) : (!llvm.ptr) -> i32
    %fmt_val = sim.fmt.dec %id specifierWidth 0 : i32
    %fmt_full = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_full
    return
  }

  hw.module @vtable_cross_func_test() {
    llhd.process {
      // Create object in constructor function
      %obj = func.call @"base::new"() : () -> !llvm.ptr

      // Call virtual methods from separate functions
      func.call @"call_get_name"(%obj) : (!llvm.ptr) -> ()
      func.call @"call_get_id"(%obj) : (!llvm.ptr) -> ()

      llhd.halt
    }
    hw.output
  }
}

// CHECK: base_obj
// CHECK: id = 42
