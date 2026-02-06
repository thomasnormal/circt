// RUN: circt-sim %s --top test | FileCheck %s
// Test that the static vtable resolution fallback works when the vtable pointer
// in an object is corrupt (contains a non-vtable address like string data).
//
// This simulates the real-world scenario where memory corruption causes a
// class instance's vtable pointer field to contain garbage (e.g., a string
// address from a nearby field overwrite), but the vtable global is intact.
//
// The fallback traces the SSA chain from func.call_indirect back through the
// unrealized_conversion_cast -> llvm.load -> llvm.getelementptr pattern to find
// the vtable global name and method index, then reads the correct function
// address directly from the vtable global's memory.
//
// Object layout: { i32 typeId, ptr vtablePtr, i32 value }
// Vtable: array of 2 pointers: [get_value, get_double]

// CHECK: corrupt_vtable_result = 42
// CHECK: corrupt_vtable_double = 84
// CHECK: [circt-sim] Simulation completed

module {
  // The vtable global with entries for two methods
  llvm.mlir.global internal @"CorruptClass::__vtable__"(#llvm.zero)
    {addr_space = 0 : i32, circt.vtable_entries = [
      [0, @"CorruptClass::get_value"],
      [1, @"CorruptClass::get_double"]
    ]} : !llvm.array<2 x ptr>

  // A dummy global that simulates a packed string - its address will be used
  // to corrupt the vtable pointer
  llvm.mlir.global internal @"__packed_string_garbage"("dump will not happen until\00")
    {addr_space = 0 : i32} : !llvm.array<27 x i8>

  // Virtual method: returns the value field
  func.func @"CorruptClass::get_value"(%this: !llvm.ptr) -> i32 {
    %val_ptr = llvm.getelementptr %this[0, 2]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"CorruptClass", (i32, ptr, i32)>
    %val = llvm.load %val_ptr : !llvm.ptr -> i32
    return %val : i32
  }

  // Virtual method: returns the value * 2
  func.func @"CorruptClass::get_double"(%this: !llvm.ptr) -> i32 {
    %val_ptr = llvm.getelementptr %this[0, 2]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"CorruptClass", (i32, ptr, i32)>
    %val = llvm.load %val_ptr : !llvm.ptr -> i32
    %two = arith.constant 2 : i32
    %result = arith.muli %val, %two : i32
    return %result : i32
  }

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^start
    ^start:
      %one = arith.constant 1 : i64
      %c42 = arith.constant 42 : i32

      // Allocate an object
      %obj = llvm.alloca %one x !llvm.struct<"CorruptClass", (i32, ptr, i32)>
          : (i64) -> !llvm.ptr

      // Store value field = 42
      %val_ptr = llvm.getelementptr %obj[0, 2]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"CorruptClass", (i32, ptr, i32)>
      llvm.store %c42, %val_ptr : i32, !llvm.ptr

      // CORRUPT the vtable pointer: store a string global address instead of
      // the real vtable. This simulates memory corruption.
      %garbage = llvm.mlir.addressof @"__packed_string_garbage"
          : !llvm.ptr
      %vt_ptr = llvm.getelementptr %obj[0, 1]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"CorruptClass", (i32, ptr, i32)>
      llvm.store %garbage, %vt_ptr : !llvm.ptr, !llvm.ptr

      // Virtual method call #1 (method index 0): load vtable pointer (will be
      // the garbage string address), GEP into "vtable"[0], load function pointer
      // (will read string bytes as a pointer), cast and call_indirect.
      // The static fallback should resolve this to CorruptClass::get_value.
      %vtable_ptr = llvm.load %vt_ptr : !llvm.ptr -> !llvm.ptr
      %method_ptr = llvm.getelementptr %vtable_ptr[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %func_ptr = llvm.load %method_ptr : !llvm.ptr -> !llvm.ptr
      %casted = builtin.unrealized_conversion_cast %func_ptr
          : !llvm.ptr to (!llvm.ptr) -> i32
      %result = func.call_indirect %casted(%obj) : (!llvm.ptr) -> i32

      %lit1 = sim.fmt.literal "corrupt_vtable_result = "
      %v1 = sim.fmt.dec %result signed : i32
      %nl1 = sim.fmt.literal "\0A"
      %fmt1 = sim.fmt.concat (%lit1, %v1, %nl1)
      sim.proc.print %fmt1

      // Virtual method call #2 (method index 1): same pattern but for get_double
      %method_ptr2 = llvm.getelementptr %vtable_ptr[0, 1]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %func_ptr2 = llvm.load %method_ptr2 : !llvm.ptr -> !llvm.ptr
      %casted2 = builtin.unrealized_conversion_cast %func_ptr2
          : !llvm.ptr to (!llvm.ptr) -> i32
      %result2 = func.call_indirect %casted2(%obj) : (!llvm.ptr) -> i32

      %lit2 = sim.fmt.literal "corrupt_vtable_double = "
      %v2 = sim.fmt.dec %result2 signed : i32
      %nl2 = sim.fmt.literal "\0A"
      %fmt2 = sim.fmt.concat (%lit2, %v2, %nl2)
      sim.proc.print %fmt2

      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}
