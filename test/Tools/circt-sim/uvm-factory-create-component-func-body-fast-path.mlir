// RUN: circt-sim %s | FileCheck %s

// Verify function-body fast paths for:
//   - uvm_component_registry_*::initialize (native registration cache)
//   - uvm_default_factory::create_component_by_name
// If either fast path does not fire, create_component_by_name returns null.

module {
  llvm.mlir.global internal constant @type_name_global("my_type") {
    addr_space = 0 : i32
  }

  llvm.mlir.global internal @"uvm_pkg::uvm_component_registry_1::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [1, @"uvm_pkg::uvm_component_registry_1::create_component"],
      [2, @"uvm_pkg::uvm_component_registry_1::get_type_name"],
      [3, @"uvm_pkg::uvm_component_registry_1::initialize"]
    ]
  } : !llvm.array<4 x ptr>

  func.func private @"uvm_pkg::uvm_component_registry_1::get_type_name"(%arg0: !llvm.ptr) -> !llvm.struct<(ptr, i64)> {
    %strAddr = llvm.mlir.addressof @type_name_global : !llvm.ptr
    %len = llvm.mlir.constant(7 : i64) : i64
    %undef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %withPtr = llvm.insertvalue %strAddr, %undef[0] : !llvm.struct<(ptr, i64)>
    %withLen = llvm.insertvalue %len, %withPtr[1] : !llvm.struct<(ptr, i64)>
    return %withLen : !llvm.struct<(ptr, i64)>
  }

  func.func private @"uvm_pkg::uvm_component_registry_1::create_component"(%arg0: !llvm.ptr, %arg1: !llvm.struct<(ptr, i64)>, %arg2: !llvm.ptr) -> !llvm.ptr {
    %addr = arith.constant 17476 : i64
    %ptr = llvm.inttoptr %addr : i64 to !llvm.ptr
    return %ptr : !llvm.ptr
  }

  func.func private @"uvm_pkg::uvm_component_registry_1::initialize"(%arg0: !llvm.ptr) {
    return
  }

  func.func private @"uvm_pkg::uvm_default_factory::create_component_by_name"(%arg0: !llvm.ptr, %arg1: !llvm.struct<(ptr, i64)>, %arg2: !llvm.struct<(ptr, i64)>, %arg3: !llvm.struct<(ptr, i64)>, %arg4: !llvm.ptr) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    return %null : !llvm.ptr
  }

  hw.module @top() {
    %fmtPrefix = sim.fmt.literal "factory fast-path = "
    %fmtNl = sim.fmt.literal "\\0A"

    llhd.process {
      %one = llvm.mlir.constant(1 : i64) : i64
      %zero64 = arith.constant 0 : i64
      %zeroPtr = llvm.inttoptr %zero64 : i64 to !llvm.ptr

      %wrapper = llvm.alloca %one x !llvm.struct<(i32, ptr)> : (i64) -> !llvm.ptr
      %classIdAddr = llvm.getelementptr %wrapper[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, ptr)>
      %classId = llvm.mlir.constant(1 : i32) : i32
      llvm.store %classId, %classIdAddr : i32, !llvm.ptr
      %vtableAddr = llvm.getelementptr %wrapper[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, ptr)>
      %vtable = llvm.mlir.addressof @"uvm_pkg::uvm_component_registry_1::__vtable__" : !llvm.ptr
      llvm.store %vtable, %vtableAddr : !llvm.ptr, !llvm.ptr

      // Register wrapper in native map via initialize fast path.
      func.call @"uvm_pkg::uvm_component_registry_1::initialize"(%wrapper) : (!llvm.ptr) -> ()

      %strAddr = llvm.mlir.addressof @type_name_global : !llvm.ptr
      %len = llvm.mlir.constant(7 : i64) : i64
      %undef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
      %typeNameWithPtr = llvm.insertvalue %strAddr, %undef[0] : !llvm.struct<(ptr, i64)>
      %typeName = llvm.insertvalue %len, %typeNameWithPtr[1] : !llvm.struct<(ptr, i64)>
      %emptyWithPtr = llvm.insertvalue %zeroPtr, %undef[0] : !llvm.struct<(ptr, i64)>
      %zeroLen = llvm.mlir.constant(0 : i64) : i64
      %empty = llvm.insertvalue %zeroLen, %emptyWithPtr[1] : !llvm.struct<(ptr, i64)>

      %created = func.call @"uvm_pkg::uvm_default_factory::create_component_by_name"(%zeroPtr, %typeName, %empty, %typeName, %zeroPtr) : (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llvm.ptr) -> !llvm.ptr

      %createdI64 = llvm.ptrtoint %created : !llvm.ptr to i64
      %expected = arith.constant 17476 : i64
      %ok = comb.icmp eq %createdI64, %expected : i64
      %okI32 = arith.extui %ok : i1 to i32
      %okFmt = sim.fmt.dec %okI32 : i32
      %line = sim.fmt.concat (%fmtPrefix, %okFmt, %fmtNl)
      sim.proc.print %line

      llhd.halt
    }

    hw.output
  }
}

// CHECK: factory fast-path = 1
