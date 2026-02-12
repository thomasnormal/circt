// RUN: not circt-sim %s --top test --timeout=2 2>&1 | FileCheck %s

// Regression: when timeout/abort triggers while executing a virtual
// dispatch, the interpreter must not report a fake internal call_indirect
// failure. We should only see the timeout message.
//
// Before the fix, abort from interpretFuncBody propagated as failure and
// produced misleading diagnostics like:
//   - "Failed in func body for process ..."
//   - "WARNING: virtual method call (func.call_indirect) failed: ..."
//
// CHECK: [circt-sim] Wall-clock timeout reached
// CHECK-NOT: Failed in func body for process
// CHECK-NOT: WARNING: virtual method call (func.call_indirect) failed

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  func.func private @"component::tick"(%self: !llvm.ptr) {
    // Keep each virtual call busy enough that timeout can trigger while
    // we're in interpretFuncBody for this callee.
    %c0 = hw.constant 0 : i32
    %c1 = hw.constant 1 : i32
    %limit = hw.constant 200000 : i32
    cf.br ^loop(%c0 : i32)
  ^loop(%i: i32):
    %cond = arith.cmpi ult, %i, %limit : i32
    cf.cond_br %cond, ^body, ^done
  ^body:
    %next = arith.addi %i, %c1 : i32
    cf.br ^loop(%next : i32)
  ^done:
    return
  }

  llvm.mlir.global internal @"component::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"component::tick"]]
  } : !llvm.array<1 x ptr>

  hw.module @test() {
    %delay = llhd.constant_time <1ns, 0d, 0e>

    llhd.process {
      %obj_size = llvm.mlir.constant(8 : i64) : i64
      %obj = llvm.call @malloc(%obj_size) : (i64) -> !llvm.ptr

      // Object layout: struct { ptr vtable }.
      %vtable_addr = llvm.mlir.addressof @"component::__vtable__" : !llvm.ptr
      %vptr_field = llvm.getelementptr %obj[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"component", (ptr)>
      llvm.store %vtable_addr, %vptr_field : !llvm.ptr, !llvm.ptr

      cf.br ^loop

    ^loop:
      %vptr = llvm.load %vptr_field : !llvm.ptr -> !llvm.ptr
      %slot0 = llvm.getelementptr %vptr[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
      %fn = builtin.unrealized_conversion_cast %fptr
          : !llvm.ptr to (!llvm.ptr) -> ()
      func.call_indirect %fn(%obj) : (!llvm.ptr) -> ()
      llhd.wait delay %delay, ^loop
    }

    hw.output
  }
}
