// RUN: circt-sim %s | FileCheck %s

// Test vtable dispatch where object pointer is stored in a global
// and loaded back in a different process. This mimics the UVM
// singleton pattern where uvm_root stores its instance in a global.

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  // Virtual method
  func.func private @"singleton::get_value"(%self: !llvm.ptr) -> i32 {
    %c99 = arith.constant 99 : i32
    return %c99 : i32
  }

  // Vtable global
  llvm.mlir.global internal @"singleton::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [0, @"singleton::get_value"]
    ]
  } : !llvm.array<1 x ptr>

  // Global to store the singleton pointer
  llvm.mlir.global internal @"singleton::instance"() {addr_space = 0 : i32} : !llvm.ptr

  // Constructor
  func.func private @"singleton::new"() -> !llvm.ptr {
    %obj_size = llvm.mlir.constant(16 : i64) : i64
    %obj = llvm.call @malloc(%obj_size) : (i64) -> !llvm.ptr

    // Object layout: !llvm.struct<"singleton", (i32, ptr)>
    %type_id = arith.constant 7 : i32
    %tid_field = llvm.getelementptr %obj[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"singleton", (i32, ptr)>
    llvm.store %type_id, %tid_field : i32, !llvm.ptr

    %vtable_addr = llvm.mlir.addressof @"singleton::__vtable__" : !llvm.ptr
    %vptr_field = llvm.getelementptr %obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"singleton", (i32, ptr)>
    llvm.store %vtable_addr, %vptr_field : !llvm.ptr, !llvm.ptr

    // Store in global
    %global_addr = llvm.mlir.addressof @"singleton::instance" : !llvm.ptr
    llvm.store %obj, %global_addr : !llvm.ptr, !llvm.ptr

    return %obj : !llvm.ptr
  }

  hw.module @vtable_global_test() {
    %fmt_prefix = sim.fmt.literal "value = "
    %fmt_nl = sim.fmt.literal "\0A"

    // Process 1: Create the singleton
    llhd.process {
      %obj = func.call @"singleton::new"() : () -> !llvm.ptr
      llhd.halt
    }

    // Process 2: Load singleton from global and call virtual method
    llhd.process {
      // Load object pointer from global
      %global_addr = llvm.mlir.addressof @"singleton::instance" : !llvm.ptr
      %obj = llvm.load %global_addr : !llvm.ptr -> !llvm.ptr

      // Virtual dispatch
      %vptr_field = llvm.getelementptr %obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"singleton", (i32, ptr)>
      %vptr = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr
      %slot_addr = llvm.getelementptr %vptr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fptr = llvm.load %slot_addr : !llvm.ptr -> !llvm.ptr
      %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr) -> i32
      %val = func.call_indirect %fn(%obj) : (!llvm.ptr) -> i32

      %fmt_val = sim.fmt.dec %val specifierWidth 0 : i32
      %fmt_full = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
      sim.proc.print %fmt_full

      llhd.halt
    }
    hw.output
  }
}

// CHECK: value = 99
