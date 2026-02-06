// RUN: circt-sim %s --max-time 100000000 2>&1 | FileCheck %s
// Test that memory loads/stores work correctly when a pointer is passed
// through a function argument. The memory block is allocated (llvm.alloca)
// in the caller and the address is passed to the callee as an !llvm.ptr
// argument. The callee must be able to load from the pointer even though
// it's a different SSA value from the original alloca result.
//
// This tests the findMemoryBlockByAddress fallback in interpretLLVMLoad.

// CHECK: RESULT=42

module {
  func.func @readval(%p: !llvm.ptr) -> i32 {
    %v = llvm.load %p : !llvm.ptr -> i32
    return %v : i32
  }

  hw.module @top() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^start
    ^start:
      %one = arith.constant 1 : i64
      %c42 = arith.constant 42 : i32
      %ptr = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
      llvm.store %c42, %ptr : i32, !llvm.ptr
      %result = func.call @readval(%ptr) : (!llvm.ptr) -> i32
      %lit = sim.fmt.literal "RESULT="
      %val = sim.fmt.dec %result signed : i32
      %nl = sim.fmt.literal "\0A"
      %fmt = sim.fmt.concat (%lit, %val, %nl)
      sim.proc.print %fmt
      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}
