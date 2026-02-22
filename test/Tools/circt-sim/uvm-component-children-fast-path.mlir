// RUN: circt-sim %s | FileCheck %s

// Verify direct-call fast paths for:
//  - uvm_component::get_num_children
//  - uvm_component::has_child
//  - uvm_component::get_child
//  - uvm_component::get_first_child
//  - uvm_component::get_next_child
// Dummy function bodies return incorrect values; test only passes if fast paths
// read the backing m_children associative array directly.

module {
  llvm.mlir.global internal constant @name_a("a")
  llvm.mlir.global internal constant @name_b("b")
  llvm.mlir.global internal constant @name_z("z")

  llvm.func @__moore_assoc_create(i32, i32) -> !llvm.ptr
  llvm.func @__moore_assoc_get_ref(!llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr

  func.func private @"uvm_pkg::uvm_component::get_num_children"(%arg0: !llvm.ptr) -> i32 {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }

  func.func private @"uvm_pkg::uvm_component::has_child"(%arg0: !llvm.ptr, %arg1: !llvm.struct<(ptr, i64)>) -> i1 {
    %false = arith.constant false
    return %false : i1
  }

  func.func private @"uvm_pkg::uvm_component::get_first_child"(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i32 {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }

  func.func private @"uvm_pkg::uvm_component::get_next_child"(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i32 {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }

  func.func private @"uvm_pkg::uvm_component::get_child"(%arg0: !llvm.ptr, %arg1: !llvm.struct<(ptr, i64)>) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    return %null : !llvm.ptr
  }

  hw.module @top() {
    %fmtPrefix = sim.fmt.literal "component-children fast-path = "
    %fmtNl = sim.fmt.literal "\\0A"

    llhd.process {
      %one = llvm.mlir.constant(1 : i64) : i64
      %zeroI64 = llvm.mlir.constant(0 : i64) : i64
      %twoI32 = arith.constant 2 : i32
      %oneI32 = arith.constant 1 : i32
      %zeroI32 = arith.constant 0 : i32
      %zeroPtr = llvm.mlir.zero : !llvm.ptr
      %eightI32 = arith.constant 8 : i32
      %zeroI32ForCreate = arith.constant 0 : i32

      %nameAPtr = llvm.mlir.addressof @name_a : !llvm.ptr
      %nameBPtr = llvm.mlir.addressof @name_b : !llvm.ptr
      %nameZPtr = llvm.mlir.addressof @name_z : !llvm.ptr
      %nameLen = llvm.mlir.constant(1 : i64) : i64
      %nameUndef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
      %nameAWithPtr = llvm.insertvalue %nameAPtr, %nameUndef[0] : !llvm.struct<(ptr, i64)>
      %nameA = llvm.insertvalue %nameLen, %nameAWithPtr[1] : !llvm.struct<(ptr, i64)>
      %nameBWithPtr = llvm.insertvalue %nameBPtr, %nameUndef[0] : !llvm.struct<(ptr, i64)>
      %nameB = llvm.insertvalue %nameLen, %nameBWithPtr[1] : !llvm.struct<(ptr, i64)>
      %nameZWithPtr = llvm.insertvalue %nameZPtr, %nameUndef[0] : !llvm.struct<(ptr, i64)>
      %nameZ = llvm.insertvalue %nameLen, %nameZWithPtr[1] : !llvm.struct<(ptr, i64)>
      %nameZeroPtr = llvm.insertvalue %zeroPtr, %nameUndef[0] : !llvm.struct<(ptr, i64)>
      %nameZero = llvm.insertvalue %zeroI64, %nameZeroPtr[1] : !llvm.struct<(ptr, i64)>

      // A raw component object backing buffer. Fast path reads m_children at
      // packed offset 95.
      %component = llvm.alloca %one x !llvm.array<104 x i8> : (i64) -> !llvm.ptr
      %childrenSlot = llvm.getelementptr %component[0, 95] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<104 x i8>
      %assoc = llvm.call @__moore_assoc_create(%zeroI32ForCreate, %eightI32) : (i32, i32) -> !llvm.ptr
      llvm.store %assoc, %childrenSlot : !llvm.ptr, !llvm.ptr

      // Child objects are represented by unique addresses.
      %childA = llvm.alloca %one x i8 : (i64) -> !llvm.ptr
      %childB = llvm.alloca %one x i8 : (i64) -> !llvm.ptr

      // Insert entries into m_children: "a" -> childA, "b" -> childB.
      %keyARef = llvm.alloca %one x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
      llvm.store %nameA, %keyARef : !llvm.struct<(ptr, i64)>, !llvm.ptr
      %childARef = llvm.call @__moore_assoc_get_ref(%assoc, %keyARef, %eightI32) : (!llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
      %childAI64 = llvm.ptrtoint %childA : !llvm.ptr to i64
      llvm.store %childAI64, %childARef : i64, !llvm.ptr

      %keyBRef = llvm.alloca %one x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
      llvm.store %nameB, %keyBRef : !llvm.struct<(ptr, i64)>, !llvm.ptr
      %childBRef = llvm.call @__moore_assoc_get_ref(%assoc, %keyBRef, %eightI32) : (!llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
      %childBI64 = llvm.ptrtoint %childB : !llvm.ptr to i64
      llvm.store %childBI64, %childBRef : i64, !llvm.ptr

      %numChildren = func.call @"uvm_pkg::uvm_component::get_num_children"(%component) : (!llvm.ptr) -> i32
      %hasA = func.call @"uvm_pkg::uvm_component::has_child"(%component, %nameA) : (!llvm.ptr, !llvm.struct<(ptr, i64)>) -> i1
      %hasZ = func.call @"uvm_pkg::uvm_component::has_child"(%component, %nameZ) : (!llvm.ptr, !llvm.struct<(ptr, i64)>) -> i1

      %iterNameRef = llvm.alloca %one x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
      llvm.store %nameZero, %iterNameRef : !llvm.struct<(ptr, i64)>, !llvm.ptr

      %firstOk = func.call @"uvm_pkg::uvm_component::get_first_child"(%component, %iterNameRef) : (!llvm.ptr, !llvm.ptr) -> i32
      %firstName = llvm.load %iterNameRef : !llvm.ptr -> !llvm.struct<(ptr, i64)>
      %firstChild = func.call @"uvm_pkg::uvm_component::get_child"(%component, %firstName) : (!llvm.ptr, !llvm.struct<(ptr, i64)>) -> !llvm.ptr

      %nextOk = func.call @"uvm_pkg::uvm_component::get_next_child"(%component, %iterNameRef) : (!llvm.ptr, !llvm.ptr) -> i32
      %nextName = llvm.load %iterNameRef : !llvm.ptr -> !llvm.struct<(ptr, i64)>
      %nextChild = func.call @"uvm_pkg::uvm_component::get_child"(%component, %nextName) : (!llvm.ptr, !llvm.struct<(ptr, i64)>) -> !llvm.ptr

      %endOk = func.call @"uvm_pkg::uvm_component::get_next_child"(%component, %iterNameRef) : (!llvm.ptr, !llvm.ptr) -> i32
      %childAFound = func.call @"uvm_pkg::uvm_component::get_child"(%component, %nameA) : (!llvm.ptr, !llvm.struct<(ptr, i64)>) -> !llvm.ptr
      %childZFound = func.call @"uvm_pkg::uvm_component::get_child"(%component, %nameZ) : (!llvm.ptr, !llvm.struct<(ptr, i64)>) -> !llvm.ptr

      %numOk = arith.cmpi eq, %numChildren, %twoI32 : i32
      %numOkI32 = arith.extui %numOk : i1 to i32
      %hasAInt = arith.extui %hasA : i1 to i32
      %hasZInt = arith.extui %hasZ : i1 to i32
      %hasZZero = arith.cmpi eq, %hasZInt, %zeroI32 : i32
      %hasZZeroI32 = arith.extui %hasZZero : i1 to i32

      %firstOkCmp = arith.cmpi eq, %firstOk, %oneI32 : i32
      %firstChildI64 = llvm.ptrtoint %firstChild : !llvm.ptr to i64
      %firstChildOk = arith.cmpi eq, %firstChildI64, %childAI64 : i64
      %firstAll = arith.andi %firstOkCmp, %firstChildOk : i1
      %firstAllI32 = arith.extui %firstAll : i1 to i32

      %nextOkCmp = arith.cmpi eq, %nextOk, %oneI32 : i32
      %nextChildI64 = llvm.ptrtoint %nextChild : !llvm.ptr to i64
      %nextChildOk = arith.cmpi eq, %nextChildI64, %childBI64 : i64
      %nextAll = arith.andi %nextOkCmp, %nextChildOk : i1
      %nextAllI32 = arith.extui %nextAll : i1 to i32
      %endOkCmp = arith.cmpi eq, %endOk, %zeroI32 : i32
      %endOkI32 = arith.extui %endOkCmp : i1 to i32

      %childAFoundI64 = llvm.ptrtoint %childAFound : !llvm.ptr to i64
      %childMatch = arith.cmpi eq, %childAFoundI64, %childAI64 : i64
      %childMatchI32 = arith.extui %childMatch : i1 to i32
      %childZFoundI64 = llvm.ptrtoint %childZFound : !llvm.ptr to i64
      %childZNull = arith.cmpi eq, %childZFoundI64, %zeroI64 : i64
      %childZNullI32 = arith.extui %childZNull : i1 to i32

      %ok0 = arith.andi %numOkI32, %hasAInt : i32
      %ok1 = arith.andi %ok0, %hasZZeroI32 : i32
      %ok2 = arith.andi %ok1, %firstAllI32 : i32
      %ok3 = arith.andi %ok2, %nextAllI32 : i32
      %ok4 = arith.andi %ok3, %endOkI32 : i32
      %ok5 = arith.andi %ok4, %childMatchI32 : i32
      %ok = arith.andi %ok5, %childZNullI32 : i32

      %lineVal = sim.fmt.dec %ok : i32
      %line = sim.fmt.concat (%fmtPrefix, %lineVal, %fmtNl)
      sim.proc.print %line

      llhd.halt
    }

    hw.output
  }
}

// CHECK: component-children fast-path = 1
