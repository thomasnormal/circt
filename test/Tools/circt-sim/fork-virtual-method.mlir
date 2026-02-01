// RUN: circt-sim %s --max-time=1000000000 | FileCheck %s
// CHECK: derived greet
// CHECK: derived greet

// Test that virtual method dispatch works correctly inside sim.fork blocks.
// This tests the fix for the bug where func.call_indirect failed to find
// the parent module in forked child processes.

module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global internal @"derived::__vtable__"(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = [[0, @"derived::greet"]]} : !llvm.array<1 x ptr>
  llvm.mlir.global internal @"base::__vtable__"(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = [[0, @"base::greet"]]} : !llvm.array<1 x ptr>
  func.func private @"base::greet"(%arg0: !llvm.ptr) {
    %0 = sim.fmt.literal "base greet\0A"
    sim.proc.print %0
    return
  }
  func.func private @"derived::greet"(%arg0: !llvm.ptr) {
    %0 = sim.fmt.literal "derived greet\0A"
    sim.proc.print %0
    return
  }
  hw.module @top() {
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %1 = llvm.mlir.addressof @"derived::__vtable__" : !llvm.ptr
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.constant(12 : i64) : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    llhd.process {
      %obj = llhd.sig %4 : !llvm.ptr
      %5 = llvm.call @malloc(%3) : (i64) -> !llvm.ptr
      %6 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"derived", (struct<"base", (i32, ptr)>)>
      llvm.store %2, %6 : i32, !llvm.ptr
      %7 = llvm.getelementptr %5[0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"derived", (struct<"base", (i32, ptr)>)>
      llvm.store %1, %7 : !llvm.ptr, !llvm.ptr
      llhd.drv %obj, %5 after %0 : !llvm.ptr
      // Direct call - should print "derived greet"
      %8 = llhd.prb %obj : !llvm.ptr
      %9 = llvm.getelementptr %8[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"base", (i32, ptr)>
      %10 = llvm.load %9 : !llvm.ptr -> !llvm.ptr
      %11 = llvm.getelementptr %10[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %12 = llvm.load %11 : !llvm.ptr -> !llvm.ptr
      %13 = builtin.unrealized_conversion_cast %12 : !llvm.ptr to (!llvm.ptr) -> ()
      func.call_indirect %13(%8) : (!llvm.ptr) -> ()
      // Fork call - should also print "derived greet"
      %14 = sim.fork {
        %15 = llhd.prb %obj : !llvm.ptr
        %16 = llvm.getelementptr %15[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"base", (i32, ptr)>
        %17 = llvm.load %16 : !llvm.ptr -> !llvm.ptr
        %18 = llvm.getelementptr %17[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
        %19 = llvm.load %18 : !llvm.ptr -> !llvm.ptr
        %20 = builtin.unrealized_conversion_cast %19 : !llvm.ptr to (!llvm.ptr) -> ()
        func.call_indirect %20(%15) : (!llvm.ptr) -> ()
        sim.fork.terminator
      }
      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
