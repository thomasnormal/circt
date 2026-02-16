// RUN: circt-sim %s | FileCheck %s

// Verify targeted report getter fast-paths:
// 1) uvm_report_object::get_report_action/get_report_verbosity_level
// 2) uvm_report_handler::get_action/get_verbosity_level via call_indirect

module {
  // Bodies return sentinel values; fast-path should override them.
  func.func private @"uvm_pkg::uvm_report_object::get_report_action"(
      %self: !llvm.ptr, %severity: i32, %id: i32) -> i32 {
    %c77 = arith.constant 77 : i32
    return %c77 : i32
  }

  func.func private @"uvm_pkg::uvm_report_object::get_report_verbosity_level"(
      %self: !llvm.ptr, %severity: i32, %id: i32) -> i32 {
    %c123 = arith.constant 123 : i32
    return %c123 : i32
  }

  func.func private @"uvm_pkg::uvm_report_handler::get_action"(
      %self: !llvm.ptr, %severity: i32, %id: i32) -> i32 {
    %c55 = arith.constant 55 : i32
    return %c55 : i32
  }

  func.func private @"uvm_pkg::uvm_report_handler::get_verbosity_level"(
      %self: !llvm.ptr, %severity: i32, %id: i32) -> i32 {
    %c66 = arith.constant 66 : i32
    return %c66 : i32
  }

  llvm.mlir.global internal @"uvm_report_fastpath::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [0, @"uvm_pkg::uvm_report_object::get_report_action"],
      [1, @"uvm_pkg::uvm_report_object::get_report_verbosity_level"],
      [2, @"uvm_pkg::uvm_report_handler::get_action"],
      [3, @"uvm_pkg::uvm_report_handler::get_verbosity_level"]
    ]
  } : !llvm.array<4 x ptr>

  hw.module @main() {
    %fmtActionInfo = sim.fmt.literal "ro action info = "
    %fmtActionErr = sim.fmt.literal "ro action error = "
    %fmtVerb = sim.fmt.literal "ro verbosity = "
    %fmtRhActionFatal = sim.fmt.literal "rh action fatal = "
    %fmtRhVerb = sim.fmt.literal "rh verbosity = "
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      %zero64 = arith.constant 0 : i64
      %self = llvm.inttoptr %zero64 : i64 to !llvm.ptr
      %sevInfo = arith.constant 0 : i32
      %sevErr = arith.constant 2 : i32
      %sevFatal = arith.constant 3 : i32
      %dummyId = arith.constant 0 : i32

      // Direct calls (func.call) hit report-object fast-paths.
      %roActionInfo = func.call @"uvm_pkg::uvm_report_object::get_report_action"(
          %self, %sevInfo, %dummyId) : (!llvm.ptr, i32, i32) -> i32
      %roActionErr = func.call @"uvm_pkg::uvm_report_object::get_report_action"(
          %self, %sevErr, %dummyId) : (!llvm.ptr, i32, i32) -> i32
      %roVerb = func.call @"uvm_pkg::uvm_report_object::get_report_verbosity_level"(
          %self, %sevInfo, %dummyId) : (!llvm.ptr, i32, i32) -> i32

      // call_indirect dispatch for report-handler fast-paths.
      %vtableAddr = llvm.mlir.addressof @"uvm_report_fastpath::__vtable__" :
          !llvm.ptr
      %slot2 = llvm.getelementptr %vtableAddr[0, 2] :
          (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x ptr>
      %fptr2 = llvm.load %slot2 : !llvm.ptr -> !llvm.ptr
      %fn2 = builtin.unrealized_conversion_cast %fptr2 :
          !llvm.ptr to (!llvm.ptr, i32, i32) -> i32
      %rhActionFatal = func.call_indirect %fn2(%self, %sevFatal, %dummyId) :
          (!llvm.ptr, i32, i32) -> i32

      %slot3 = llvm.getelementptr %vtableAddr[0, 3] :
          (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x ptr>
      %fptr3 = llvm.load %slot3 : !llvm.ptr -> !llvm.ptr
      %fn3 = builtin.unrealized_conversion_cast %fptr3 :
          !llvm.ptr to (!llvm.ptr, i32, i32) -> i32
      %rhVerb = func.call_indirect %fn3(%self, %sevInfo, %dummyId) :
          (!llvm.ptr, i32, i32) -> i32

      %roActionInfoDec = sim.fmt.dec %roActionInfo signed : i32
      %roActionInfoLine = sim.fmt.concat (%fmtActionInfo, %roActionInfoDec, %fmtNl)
      sim.proc.print %roActionInfoLine

      %roActionErrDec = sim.fmt.dec %roActionErr signed : i32
      %roActionErrLine = sim.fmt.concat (%fmtActionErr, %roActionErrDec, %fmtNl)
      sim.proc.print %roActionErrLine

      %roVerbDec = sim.fmt.dec %roVerb signed : i32
      %roVerbLine = sim.fmt.concat (%fmtVerb, %roVerbDec, %fmtNl)
      sim.proc.print %roVerbLine

      %rhActionFatalDec = sim.fmt.dec %rhActionFatal signed : i32
      %rhActionFatalLine = sim.fmt.concat (%fmtRhActionFatal, %rhActionFatalDec, %fmtNl)
      sim.proc.print %rhActionFatalLine

      %rhVerbDec = sim.fmt.dec %rhVerb signed : i32
      %rhVerbLine = sim.fmt.concat (%fmtRhVerb, %rhVerbDec, %fmtNl)
      sim.proc.print %rhVerbLine

      llhd.halt
    }
    hw.output
  }
}

// CHECK: ro action info = 1
// CHECK: ro action error = 9
// CHECK: ro verbosity = 200
// CHECK: rh action fatal = 33
// CHECK: rh verbosity = 200
