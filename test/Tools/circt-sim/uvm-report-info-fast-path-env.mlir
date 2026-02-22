// RUN: env CIRCT_SIM_FASTPATH_UVM_REPORT_INFO=1 circt-sim %s | FileCheck %s

// Verify env-gated UVM report-info suppression fast-path.
// The report function body writes 1 to @report_flag if executed.
// With CIRCT_SIM_FASTPATH_UVM_REPORT_INFO=1, call_indirect should bypass it.

module {
  llvm.mlir.global internal @report_flag(0 : i32) {addr_space = 0 : i32} : i32

  func.func private @"uvm_pkg::uvm_report_object::uvm_report_info"(
      %self: !llvm.ptr, %id: i32) {
    %g = llvm.mlir.addressof @report_flag : !llvm.ptr
    %one = arith.constant 1 : i32
    llvm.store %one, %g : i32, !llvm.ptr
    return
  }

  llvm.mlir.global internal @"uvm_report_info_fastpath::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [0, @"uvm_pkg::uvm_report_object::uvm_report_info"]
    ]
  } : !llvm.array<1 x ptr>

  hw.module @main() {
    %fmt = sim.fmt.literal "report flag = "
    %nl = sim.fmt.literal "\0A"

    llhd.process {
      %zero64 = arith.constant 0 : i64
      %self = llvm.inttoptr %zero64 : i64 to !llvm.ptr
      %id = arith.constant 0 : i32

      %vt = llvm.mlir.addressof @"uvm_report_info_fastpath::__vtable__" : !llvm.ptr
      %slot = llvm.getelementptr %vt[0, 0] :
          (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fptr = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %fn = builtin.unrealized_conversion_cast %fptr :
          !llvm.ptr to (!llvm.ptr, i32) -> ()
      func.call_indirect %fn(%self, %id) : (!llvm.ptr, i32) -> ()

      %g = llvm.mlir.addressof @report_flag : !llvm.ptr
      %val = llvm.load %g : !llvm.ptr -> i32
      %valFmt = sim.fmt.dec %val signed : i32
      %line = sim.fmt.concat (%fmt, %valFmt, %nl)
      sim.proc.print %line

      llhd.halt
    }
    hw.output
  }
}

// CHECK: report flag = 0
