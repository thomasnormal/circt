// RUN: circt-sim %s --top test | FileCheck %s
// Test that the fallback vtable dispatch mechanism resolves virtual method calls
// when the vtable pointer in an object is null (uninitialized).
//
// Scenario: An object is created via malloc and its vtable pointer field is
// left as zero. A virtual method call loads the vtable pointer (0), GEPs into
// it, and gets X for the function pointer. The fallback mechanism traces back
// through the SSA chain to find the vtable global name and method index, then
// resolves the function from the global's memory directly.
//
// Object layout: { i32 typeId, ptr vtablePtr, i32 value }
// Vtable: array of 2 pointers: [get_value, get_double]

// CHECK: fallback_result = 42
// CHECK: [circt-sim] Simulation completed

module {
  // The vtable global with entries for two methods
  llvm.mlir.global internal @"TestClass::__vtable__"(#llvm.zero)
    {addr_space = 0 : i32, circt.vtable_entries = [
      [0, @"TestClass::get_value"],
      [1, @"TestClass::get_double"]
    ]} : !llvm.array<2 x ptr>

  // Virtual method: returns the value field
  func.func @"TestClass::get_value"(%this: !llvm.ptr) -> i32 {
    %val_ptr = llvm.getelementptr %this[0, 2]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"TestClass", (i32, ptr, i32)>
    %val = llvm.load %val_ptr : !llvm.ptr -> i32
    return %val : i32
  }

  // Virtual method: returns the value * 2
  func.func @"TestClass::get_double"(%this: !llvm.ptr) -> i32 {
    %val_ptr = llvm.getelementptr %this[0, 2]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"TestClass", (i32, ptr, i32)>
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

      // Allocate an object via alloca (simulates malloc)
      %obj = llvm.alloca %one x !llvm.struct<"TestClass", (i32, ptr, i32)>
          : (i64) -> !llvm.ptr

      // Store value field = 42, but DO NOT store the vtable pointer
      // (leave it as zero to trigger the fallback)
      %val_ptr = llvm.getelementptr %obj[0, 2]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"TestClass", (i32, ptr, i32)>
      llvm.store %c42, %val_ptr : i32, !llvm.ptr

      // Zero-initialize the vtable pointer field explicitly
      %null = llvm.mlir.zero : !llvm.ptr
      %vt_ptr = llvm.getelementptr %obj[0, 1]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"TestClass", (i32, ptr, i32)>
      llvm.store %null, %vt_ptr : !llvm.ptr, !llvm.ptr

      // Virtual method call: load vtable pointer (will be 0), then
      // GEP into vtable[0], load function pointer (will be X),
      // cast and call_indirect. The fallback should resolve this
      // to TestClass::get_value by tracing the SSA chain.
      %vtable_ptr = llvm.load %vt_ptr : !llvm.ptr -> !llvm.ptr
      %method_ptr = llvm.getelementptr %vtable_ptr[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %func_ptr = llvm.load %method_ptr : !llvm.ptr -> !llvm.ptr
      %casted = builtin.unrealized_conversion_cast %func_ptr
          : !llvm.ptr to (!llvm.ptr) -> i32
      %result = func.call_indirect %casted(%obj) : (!llvm.ptr) -> i32

      %lit = sim.fmt.literal "fallback_result = "
      %v = sim.fmt.dec %result signed : i32
      %nl = sim.fmt.literal "\0A"
      %fmt = sim.fmt.concat (%lit, %v, %nl)
      sim.proc.print %fmt

      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}
