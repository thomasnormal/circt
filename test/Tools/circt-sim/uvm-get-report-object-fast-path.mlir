// RUN: env CIRCT_SIM_FASTPATH_UVM_GET_REPORT_OBJECT=1 circt-sim %s | FileCheck %s
// RUN: env CIRCT_UVM_ARGS='+UVM_TESTNAME=smoke' circt-sim %s | FileCheck %s

// Verify that uvm_get_report_object fast-path returns `self` for both direct
// and call_indirect dispatch, even when the callee body returns null.

module {
  // If this body executes, it returns null. The fast-path must return %self.
  func.func private @"uvm_pkg::uvm_report_object::uvm_get_report_object"(
      %self: !llvm.ptr) -> !llvm.ptr {
    %zero64 = arith.constant 0 : i64
    %null = llvm.inttoptr %zero64 : i64 to !llvm.ptr
    return %null : !llvm.ptr
  }

  llvm.mlir.global internal @"uvm_report_object_fastpath::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"uvm_pkg::uvm_report_object::uvm_get_report_object"]]
  } : !llvm.array<1 x ptr>

  hw.module @main() {
    %fmtDirect = sim.fmt.literal "direct self-match = "
    %fmtIndirect = sim.fmt.literal "indirect self-match = "
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      %self64 = arith.constant 4096 : i64
      %self = llvm.inttoptr %self64 : i64 to !llvm.ptr

      %direct = func.call @"uvm_pkg::uvm_report_object::uvm_get_report_object"(%self) :
          (!llvm.ptr) -> !llvm.ptr
      %eqDirect = llvm.icmp "eq" %direct, %self : !llvm.ptr
      %eqDirectI32 = arith.extui %eqDirect : i1 to i32

      %vtableAddr = llvm.mlir.addressof @"uvm_report_object_fastpath::__vtable__" :
          !llvm.ptr
      %slot0 = llvm.getelementptr %vtableAddr[0, 0] :
          (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fptr0 = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
      %fn0 = builtin.unrealized_conversion_cast %fptr0 :
          !llvm.ptr to (!llvm.ptr) -> !llvm.ptr
      %indirect = func.call_indirect %fn0(%self) : (!llvm.ptr) -> !llvm.ptr
      %eqIndirect = llvm.icmp "eq" %indirect, %self : !llvm.ptr
      %eqIndirectI32 = arith.extui %eqIndirect : i1 to i32

      %directDec = sim.fmt.dec %eqDirectI32 signed : i32
      %directLine = sim.fmt.concat (%fmtDirect, %directDec, %fmtNl)
      sim.proc.print %directLine

      %indirectDec = sim.fmt.dec %eqIndirectI32 signed : i32
      %indirectLine = sim.fmt.concat (%fmtIndirect, %indirectDec, %fmtNl)
      sim.proc.print %indirectLine

      llhd.halt
    }
    hw.output
  }
}

// CHECK: direct self-match = 1
// CHECK: indirect self-match = 1
