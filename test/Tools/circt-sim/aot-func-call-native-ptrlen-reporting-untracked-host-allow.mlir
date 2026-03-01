// RUN: env CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING=1 circt-compile %s -o %t.so
// RUN: env CIRCT_AOT_ALLOW_NATIVE_UVM_REPORTING=1 circt-sim %s --top ptrlen_reporting_untracked_host_allow_test --mode=compile --compiled=%t.so --aot-stats 2>&1 | FileCheck %s

// Regression/perf guard: when reporting opt-in is enabled, reporting query
// helpers should stay on native func.call dispatch even when the ptr+len tail
// argument points at an untracked host-like address. This shape is common in
// AVIP startup report-id lookups.

module {
  func.func private @"uvm_pkg::uvm_report_object::get_report_action"(
      %self: !llvm.ptr, %severity: i2,
      %id: !llvm.struct<(ptr, i64)>) -> i32 {
    %len = llvm.extractvalue %id[1] : !llvm.struct<(ptr, i64)>
    %sev64 = arith.extui %severity : i2 to i64
    %sum = arith.addi %len, %sev64 : i64
    %sum32 = arith.trunci %sum : i64 to i32
    return %sum32 : i32
  }

  hw.module @ptrlen_reporting_untracked_host_allow_test() {
    %fmtPrefix = sim.fmt.literal "result="
    %fmtNl = sim.fmt.literal "\0A"
    llhd.process {
      %one = llvm.mlir.constant(1 : i64) : i64
      %self = llvm.alloca %one x i8 : (i64) -> !llvm.ptr
      %sev = arith.constant 1 : i2
      // Host-like pointer value intentionally left untracked by interpreter
      // memory maps; used to reproduce reporting ptr+len demotion behavior.
      %idPtrI64 = llvm.mlir.constant(205015232 : i64) : i64
      %idPtr = llvm.inttoptr %idPtrI64 : i64 to !llvm.ptr
      %idLen = llvm.mlir.constant(11 : i64) : i64
      %id0 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
      %id1 = llvm.insertvalue %idPtr, %id0[0] : !llvm.struct<(ptr, i64)>
      %id2 = llvm.insertvalue %idLen, %id1[1] : !llvm.struct<(ptr, i64)>
      %r = func.call @"uvm_pkg::uvm_report_object::get_report_action"(
          %self, %sev, %id2) :
          (!llvm.ptr, i2, !llvm.struct<(ptr, i64)>) -> i32
      %fmtDec = sim.fmt.dec %r signed : i32
      %fmtOut = sim.fmt.concat (%fmtPrefix, %fmtDec, %fmtNl)
      sim.proc.print %fmtOut
      llhd.halt
    }
    hw.output
  }
}

// CHECK: direct_calls_native:              1
// CHECK: direct_calls_interpreted:         0
// CHECK: result=12
