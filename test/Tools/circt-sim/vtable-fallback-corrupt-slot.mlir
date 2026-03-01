// RUN: circt-sim %s --top test | FileCheck %s
// Regression: if a runtime vtable slot contains an unmapped pointer, static
// fallback should still resolve by vtable metadata (slot index) instead of
// silently returning zero.
//
// CHECK: slot_corrupt_result = 42
// CHECK-NOT: WARNING: virtual method call (func.call_indirect) failed

module {
  llvm.mlir.global internal @"SlotClass::__vtable__"(#llvm.zero)
      {addr_space = 0 : i32, circt.vtable_entries = [[0, @"SlotClass::get_value"]]}
      : !llvm.array<1 x ptr>

  // Non-function global used to corrupt the method slot at runtime.
  llvm.mlir.global internal @"__garbage_global"("not_a_func\00")
      {addr_space = 0 : i32}
      : !llvm.array<11 x i8>

  func.func @"SlotClass::get_value"(%this: !llvm.ptr) -> i32 {
    %val_ptr = llvm.getelementptr %this[0, 1]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"SlotClass", (ptr, i32)>
    %val = llvm.load %val_ptr : !llvm.ptr -> i32
    return %val : i32
  }

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      %one = arith.constant 1 : i64
      %c42 = arith.constant 42 : i32

      %obj = llvm.alloca %one x !llvm.struct<"SlotClass", (ptr, i32)>
          : (i64) -> !llvm.ptr

      %vtable_addr = llvm.mlir.addressof @"SlotClass::__vtable__" : !llvm.ptr
      %vt_field = llvm.getelementptr %obj[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"SlotClass", (ptr, i32)>
      llvm.store %vtable_addr, %vt_field : !llvm.ptr, !llvm.ptr

      %val_field = llvm.getelementptr %obj[0, 1]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"SlotClass", (ptr, i32)>
      llvm.store %c42, %val_field : i32, !llvm.ptr

      // Corrupt vtable slot 0: store an unmapped non-function pointer value.
      %garbage = llvm.mlir.addressof @"__garbage_global" : !llvm.ptr
      %slot0 = llvm.getelementptr %vtable_addr[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      llvm.store %garbage, %slot0 : !llvm.ptr, !llvm.ptr

      // Virtual dispatch should still resolve via vtable metadata.
      %vt = llvm.load %vt_field : !llvm.ptr -> !llvm.ptr
      %m0 = llvm.getelementptr %vt[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fp = llvm.load %m0 : !llvm.ptr -> !llvm.ptr
      %casted = builtin.unrealized_conversion_cast %fp : !llvm.ptr to (!llvm.ptr) -> i32
      %result = func.call_indirect %casted(%obj) : (!llvm.ptr) -> i32

      %lit = sim.fmt.literal "slot_corrupt_result = "
      %v = sim.fmt.dec %result signed : i32
      %nl = sim.fmt.literal "\0A"
      %msg = sim.fmt.concat (%lit, %v, %nl)
      sim.proc.print %msg

      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}
