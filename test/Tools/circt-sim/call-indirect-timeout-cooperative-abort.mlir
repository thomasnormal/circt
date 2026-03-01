// RUN: bash -c 'timeout --signal=KILL 8s circt-sim %s --top top --timeout=1 --resource-guard=false > %t.log 2>&1; cat %t.log; exit 0' | FileCheck %s

// Regression: wall-clock timeout must stop deep call_indirect traffic even
// when each frame does too little local work to hit 16K-op abort checks.
// Before the fix, this printed "Wall-clock timeout reached (global guard)"
// and then required SIGKILL from the outer timeout wrapper.
//
// CHECK: [circt-sim] Wall-clock timeout reached
// CHECK: [circt-sim] Interrupt signal received
// CHECK: [circt-sim] Simulation finished with exit code 1

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  func.func private @"component::tick"(%self: !llvm.ptr) {
    %vptr_field = llvm.getelementptr %self[0, 0]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"component", (ptr)>
    %vptr = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr
    %slot0 = llvm.getelementptr %vptr[0, 0]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fptr
        : !llvm.ptr to (!llvm.ptr) -> ()
    func.call_indirect %fn(%self) : (!llvm.ptr) -> ()
    return
  }

  func.func private @"component::spin"(%self: !llvm.ptr) {
    %vptr_field = llvm.getelementptr %self[0, 0]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"component", (ptr)>
    %vptr = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr
    %slot0 = llvm.getelementptr %vptr[0, 0]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fptr
        : !llvm.ptr to (!llvm.ptr) -> ()
    cf.br ^loop
  ^loop:
    func.call_indirect %fn(%self) : (!llvm.ptr) -> ()
    cf.br ^loop
  }

  llvm.mlir.global internal @"component::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"component::tick"]]
  } : !llvm.array<1 x ptr>

  hw.module @top() {
    llhd.process {
      %obj_size = llvm.mlir.constant(8 : i64) : i64
      %obj = llvm.call @malloc(%obj_size) : (i64) -> !llvm.ptr
      %vtable_addr = llvm.mlir.addressof @"component::__vtable__" : !llvm.ptr
      %vptr_field = llvm.getelementptr %obj[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"component", (ptr)>
      llvm.store %vtable_addr, %vptr_field : !llvm.ptr, !llvm.ptr
      func.call @"component::spin"(%obj) : (!llvm.ptr) -> ()
      llhd.halt
    }

    hw.output
  }
}
