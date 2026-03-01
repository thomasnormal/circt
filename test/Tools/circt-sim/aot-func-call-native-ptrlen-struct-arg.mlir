// RUN: circt-compile %s -o %t.so
// RUN: circt-sim %s --top ptrlen_abi_test --mode=compile --compiled=%t.so --aot-stats 2>&1 | FileCheck %s

// Regression/perf guard: native direct func.call dispatch should support
// the hot reporting-style ABI shape (u64-like, scalar, struct<(ptr, i64)>).
// Before this, compile-mode dispatch fell back with [abi=...] for such calls.

module {
  func.func private @report_ptrlen_abi_func(
      %self: !llvm.ptr, %severity: i2,
      %id: !llvm.struct<(ptr, i64)>) -> i32 {
    %len = llvm.extractvalue %id[1] : !llvm.struct<(ptr, i64)>
    %sev64 = arith.extui %severity : i2 to i64
    %sum = arith.addi %len, %sev64 : i64
    %sum32 = arith.trunci %sum : i64 to i32
    return %sum32 : i32
  }

  hw.module @ptrlen_abi_test() {
    %fmtPrefix = sim.fmt.literal "result="
    %fmtNl = sim.fmt.literal "\0A"
    llhd.process {
      %zero64 = arith.constant 0 : i64
      %self = llvm.inttoptr %zero64 : i64 to !llvm.ptr
      %sev = arith.constant 2 : i2
      %len = llvm.mlir.constant(123 : i64) : i64
      %id0 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
      %id1 = llvm.insertvalue %self, %id0[0] : !llvm.struct<(ptr, i64)>
      %id2 = llvm.insertvalue %len, %id1[1] : !llvm.struct<(ptr, i64)>
      %r = func.call @report_ptrlen_abi_func(
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
// CHECK: result=125
