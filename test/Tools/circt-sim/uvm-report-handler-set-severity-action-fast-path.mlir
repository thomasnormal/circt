// RUN: circt-sim %s | FileCheck %s
//
// Verify report_handler::set_severity_action is suppressed as a no-op for both
// func.call and call_indirect dispatch.

module {
  llvm.mlir.global internal @counter(0 : i32) : i32

  // If this executes, it increments @counter.
  func.func private @"uvm_pkg::uvm_report_handler::set_severity_action"() {
    %addr = llvm.mlir.addressof @counter : !llvm.ptr
    %cur = llvm.load %addr : !llvm.ptr -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %cur, %one : i32
    llvm.store %next, %addr : i32, !llvm.ptr
    return
  }

  llvm.mlir.global internal @"uvm_rh_set_severity_action::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [0, @"uvm_pkg::uvm_report_handler::set_severity_action"]
    ]
  } : !llvm.array<1 x ptr>

  func.func private @drive_direct() {
    func.call @"uvm_pkg::uvm_report_handler::set_severity_action"() : () -> ()
    return
  }

  func.func private @drive_indirect(%fn: () -> ()) {
    func.call_indirect %fn() : () -> ()
    return
  }

  hw.module @top() {
    %counterPrefix = sim.fmt.literal "counter = "
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      %vtableAddr = llvm.mlir.addressof @"uvm_rh_set_severity_action::__vtable__" :
          !llvm.ptr
      %slot0 = llvm.getelementptr %vtableAddr[0, 0] :
          (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fptr0 = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
      %fn = builtin.unrealized_conversion_cast %fptr0 : !llvm.ptr to () -> ()

      func.call @drive_direct() : () -> ()
      func.call @drive_direct() : () -> ()
      func.call @drive_indirect(%fn) : (() -> ()) -> ()
      func.call @drive_indirect(%fn) : (() -> ()) -> ()

      %addr = llvm.mlir.addressof @counter : !llvm.ptr
      %cur = llvm.load %addr : !llvm.ptr -> i32
      %curFmt = sim.fmt.dec %cur signed : i32
      %line = sim.fmt.concat (%counterPrefix, %curFmt, %fmtNl)
      sim.proc.print %line

      llhd.halt
    }
    hw.output
  }
}

// CHECK: counter = 0
