// RUN: circt-sim %s | FileCheck %s

// Verify duplicate uvm_phase::add calls are elided by function-body fast path
// when optional predecessor/successor arguments are all null.

module {
  llvm.mlir.global internal @g_add_calls(0 : i32) : i32

  func.func private @"uvm_pkg::uvm_phase::add"(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr) {
    %g = llvm.mlir.addressof @g_add_calls : !llvm.ptr
    %old = llvm.load %g : !llvm.ptr -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %old, %one : i32
    llvm.store %next, %g : i32, !llvm.ptr
    return
  }

  hw.module @top() {
    %fmtPrefix = sim.fmt.literal "phase-add calls="
    %fmtNl = sim.fmt.literal "\\0A"

    llhd.process {
      %a0 = arith.constant 4096 : i64
      %a1 = arith.constant 8192 : i64
      %zero64 = arith.constant 0 : i64
      %p0 = llvm.inttoptr %a0 : i64 to !llvm.ptr
      %p1 = llvm.inttoptr %a1 : i64 to !llvm.ptr
      %null = llvm.inttoptr %zero64 : i64 to !llvm.ptr

      func.call @"uvm_pkg::uvm_phase::add"(%p0, %p1, %null, %null, %null, %null, %null) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
      func.call @"uvm_pkg::uvm_phase::add"(%p0, %p1, %null, %null, %null, %null, %null) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()

      %g = llvm.mlir.addressof @g_add_calls : !llvm.ptr
      %calls = llvm.load %g : !llvm.ptr -> i32
      %countFmt = sim.fmt.dec %calls signed : i32
      %line = sim.fmt.concat (%fmtPrefix, %countFmt, %fmtNl)
      sim.proc.print %line

      llhd.halt
    }

    hw.output
  }
}

// CHECK: phase-add calls=1
