// RUN: circt-sim %s | FileCheck %s
//
// Verify global-root wrapper interception:
//   - m_uvm_get_root()
//   - uvm_pkg::uvm_get_report_object()
// Both should return uvm_root::m_inst once initialized, even if function
// bodies would otherwise return null.

module {
  llvm.mlir.global internal @"uvm_pkg::uvm_pkg::uvm_root::m_inst"(#llvm.zero) {
    addr_space = 0 : i32
  } : !llvm.ptr

  func.func private @m_uvm_get_root() -> !llvm.ptr {
    %zero64 = arith.constant 0 : i64
    %null = llvm.inttoptr %zero64 : i64 to !llvm.ptr
    return %null : !llvm.ptr
  }

  func.func private @"uvm_pkg::uvm_get_report_object"() -> !llvm.ptr {
    %zero64 = arith.constant 0 : i64
    %null = llvm.inttoptr %zero64 : i64 to !llvm.ptr
    return %null : !llvm.ptr
  }

  hw.module @top() {
    %fmtRoot = sim.fmt.literal "root fast-path = "
    %fmtReport = sim.fmt.literal "report-object fast-path = "
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      %inst = llvm.mlir.addressof @"uvm_pkg::uvm_pkg::uvm_root::m_inst" : !llvm.ptr
      %root64 = arith.constant 4660 : i64
      %root = llvm.inttoptr %root64 : i64 to !llvm.ptr
      llvm.store %root, %inst : !llvm.ptr, !llvm.ptr

      %r0 = func.call @m_uvm_get_root() : () -> !llvm.ptr
      %eq0 = llvm.icmp "eq" %r0, %root : !llvm.ptr
      %eq0i32 = arith.extui %eq0 : i1 to i32

      %r1 = func.call @"uvm_pkg::uvm_get_report_object"() : () -> !llvm.ptr
      %eq1 = llvm.icmp "eq" %r1, %root : !llvm.ptr
      %eq1i32 = arith.extui %eq1 : i1 to i32

      %rootDec = sim.fmt.dec %eq0i32 signed : i32
      %rootLine = sim.fmt.concat (%fmtRoot, %rootDec, %fmtNl)
      sim.proc.print %rootLine

      %reportDec = sim.fmt.dec %eq1i32 signed : i32
      %reportLine = sim.fmt.concat (%fmtReport, %reportDec, %fmtNl)
      sim.proc.print %reportLine

      llhd.halt
    }
    hw.output
  }
}

// CHECK: root fast-path = 1
// CHECK: report-object fast-path = 1
