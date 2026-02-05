// RUN: circt-sim %s | FileCheck %s

// Test vtable dispatch with exact UVM struct nesting pattern:
//   uvm_void:      struct<(i32, ptr)>     -- typeId, vtable_ptr
//   uvm_object:    struct<(uvm_void, struct<(ptr, i64)>, i32)>
//   uvm_phase:     struct<(uvm_object, i32, ptr, ptr)>
//
// The vtable pointer is at [0, 0, 0, 1] from uvm_phase.
// This tests deeply nested GEP indexing through structs.

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  func.func private @"phase::get_name"(%self: !llvm.ptr) -> !llvm.struct<(ptr, i64)> {
    %msg = sim.fmt.literal "build_phase\0A"
    sim.proc.print %msg
    %null = llvm.mlir.zero : !llvm.ptr
    %zero = llvm.mlir.constant(0 : i64) : i64
    %result = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %r1 = llvm.insertvalue %null, %result[0] : !llvm.struct<(ptr, i64)>
    %r2 = llvm.insertvalue %zero, %r1[1] : !llvm.struct<(ptr, i64)>
    return %r2 : !llvm.struct<(ptr, i64)>
  }

  func.func private @"phase::get_type"(%self: !llvm.ptr) -> i32 {
    %c5 = arith.constant 5 : i32
    return %c5 : i32
  }

  // Large vtable (34 slots, like UVM phase)
  llvm.mlir.global internal @"phase::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [5, @"phase::get_name"],
      [10, @"phase::get_type"]
    ]
  } : !llvm.array<34 x ptr>

  // Constructor: create a phase object with UVM-like nesting
  func.func private @"phase::new"() -> !llvm.ptr {
    // Compute exact size: uvm_phase layout
    // uvm_void: i32(4) + ptr(8) = 12 (or 16 with alignment)
    // uvm_object: uvm_void + struct<(ptr,i64)>(16) + i32(4) = ~36
    // uvm_phase: uvm_object + i32(4) + ptr(8) + ptr(8) = ~56
    // Use generous size to account for any padding
    %sz = llvm.mlir.constant(128 : i64) : i64
    %obj = llvm.call @malloc(%sz) : (i64) -> !llvm.ptr

    // Store type_id at [0, 0, 0, 0] (uvm_phase -> uvm_object -> uvm_void -> i32)
    %tid_field = llvm.getelementptr %obj[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"uvm_phase", (struct<"uvm_object", (struct<"uvm_void", (i32, ptr)>, struct<(ptr, i64)>, i32)>, i32, ptr, ptr)>
    %tid = arith.constant 42 : i32
    llvm.store %tid, %tid_field : i32, !llvm.ptr

    // Store vtable pointer at [0, 0, 0, 1] (uvm_phase -> uvm_object -> uvm_void -> ptr)
    %vt = llvm.mlir.addressof @"phase::__vtable__" : !llvm.ptr
    %vptr_field = llvm.getelementptr %obj[0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"uvm_phase", (struct<"uvm_object", (struct<"uvm_void", (i32, ptr)>, struct<(ptr, i64)>, i32)>, i32, ptr, ptr)>
    llvm.store %vt, %vptr_field : !llvm.ptr, !llvm.ptr

    return %obj : !llvm.ptr
  }

  // Dispatch function - loads vtable from deeply nested field
  func.func private @"dispatch_get_name"(%phase: !llvm.ptr) {
    // Load vtable pointer from [0, 0, 0, 1]
    %vptr_field = llvm.getelementptr %phase[0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"uvm_phase", (struct<"uvm_object", (struct<"uvm_void", (i32, ptr)>, struct<(ptr, i64)>, i32)>, i32, ptr, ptr)>
    %vptr = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr

    // Index into vtable at slot 5 (get_name)
    %slot = llvm.getelementptr %vptr[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<34 x ptr>
    %fptr = llvm.load %slot : !llvm.ptr -> !llvm.ptr

    // Cast and call
    %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr) -> !llvm.struct<(ptr, i64)>
    %result = func.call_indirect %fn(%phase) : (!llvm.ptr) -> !llvm.struct<(ptr, i64)>
    return
  }

  func.func private @"dispatch_get_type"(%phase: !llvm.ptr) {
    %fmt_prefix = sim.fmt.literal "type = "
    %fmt_nl = sim.fmt.literal "\0A"

    %vptr_field = llvm.getelementptr %phase[0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"uvm_phase", (struct<"uvm_object", (struct<"uvm_void", (i32, ptr)>, struct<(ptr, i64)>, i32)>, i32, ptr, ptr)>
    %vptr = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr

    %slot = llvm.getelementptr %vptr[0, 10] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<34 x ptr>
    %fptr = llvm.load %slot : !llvm.ptr -> !llvm.ptr

    %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr) -> i32
    %val = func.call_indirect %fn(%phase) : (!llvm.ptr) -> i32

    %fmt_val = sim.fmt.dec %val specifierWidth 0 : i32
    %fmt_full = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_full
    return
  }

  hw.module @vtable_uvm_nesting_test() {
    llhd.process {
      %phase = func.call @"phase::new"() : () -> !llvm.ptr
      func.call @"dispatch_get_name"(%phase) : (!llvm.ptr) -> ()
      func.call @"dispatch_get_type"(%phase) : (!llvm.ptr) -> ()
      llhd.halt
    }
    hw.output
  }
}

// CHECK: build_phase
// CHECK: type = 5
