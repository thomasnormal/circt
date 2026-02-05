// RUN: circt-sim %s | FileCheck %s

// Test vtable dispatch where object is created in a global constructor
// (LLVM function) that uses func.call to call a func.func constructor.
// This exercises the exact pattern used in UVM where LLVM global_ctors
// contain func.call ops to class constructors.

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  // Virtual method
  func.func private @"root::get_id"(%self: !llvm.ptr) -> i32 {
    %c123 = arith.constant 123 : i32
    return %c123 : i32
  }

  // Vtable global
  llvm.mlir.global internal @"root::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [0, @"root::get_id"]
    ]
  } : !llvm.array<1 x ptr>

  // Global to hold the singleton instance
  llvm.mlir.global internal @"root::instance"(0 : i64) {addr_space = 0 : i32} : !llvm.ptr

  // Constructor as func.func (this is how MooreToCore emits constructors)
  func.func private @"root::new"() -> !llvm.ptr {
    %sz = llvm.mlir.constant(16 : i64) : i64
    %obj = llvm.call @malloc(%sz) : (i64) -> !llvm.ptr

    // Store type_id
    %tid = arith.constant 1 : i32
    %tid_field = llvm.getelementptr %obj[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"root", (i32, ptr)>
    llvm.store %tid, %tid_field : i32, !llvm.ptr

    // Store vtable pointer
    %vt = llvm.mlir.addressof @"root::__vtable__" : !llvm.ptr
    %vptr_field = llvm.getelementptr %obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"root", (i32, ptr)>
    llvm.store %vt, %vptr_field : !llvm.ptr, !llvm.ptr

    return %obj : !llvm.ptr
  }

  // LLVM global constructor that uses func.call to call the constructor
  llvm.func internal @"__ctor_root"() {
    %obj = func.call @"root::new"() : () -> !llvm.ptr
    // Store in global
    %global = llvm.mlir.addressof @"root::instance" : !llvm.ptr
    llvm.store %obj, %global : !llvm.ptr, !llvm.ptr
    llvm.return
  }

  // Register as global constructor
  llvm.mlir.global_ctors ctors = [@"__ctor_root"], priorities = [65535 : i32], data = [#llvm.zero]

  hw.module @vtable_global_ctor_test() {
    %fmt_prefix = sim.fmt.literal "ctor id = "
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      // Load singleton from global (set by global constructor)
      %global = llvm.mlir.addressof @"root::instance" : !llvm.ptr
      %obj = llvm.load %global : !llvm.ptr -> !llvm.ptr

      // Virtual dispatch
      %vptr_field = llvm.getelementptr %obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"root", (i32, ptr)>
      %vptr = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr
      %slot = llvm.getelementptr %vptr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fptr = llvm.load %slot : !llvm.ptr -> !llvm.ptr
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

// CHECK: ctor id = 123
