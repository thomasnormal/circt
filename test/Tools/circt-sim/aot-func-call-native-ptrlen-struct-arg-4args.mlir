// RUN: circt-compile %s -o %t.so
// RUN: circt-sim %s --top ptrlen_abi_test_4args --mode=compile --compiled=%t.so --aot-stats 2>&1 | FileCheck %s

// Regression/perf guard: direct native func.call dispatch should also support
// ptr+len aggregate ABI when the callee has additional scalar prefix args.
// This mirrors APB hotspot shape:
//   (!llvm.ptr, i32, i2, !llvm.struct<(ptr, i64)>) -> i32.

module {
  func.func private @report_ptrlen_abi_func4(
      %self: !llvm.ptr, %verb: i32, %severity: i2,
      %id: !llvm.struct<(ptr, i64)>) -> i32 {
    %len = llvm.extractvalue %id[1] : !llvm.struct<(ptr, i64)>
    %verb64 = arith.extui %verb : i32 to i64
    %sev64 = arith.extui %severity : i2 to i64
    %sum0 = arith.addi %len, %verb64 : i64
    %sum1 = arith.addi %sum0, %sev64 : i64
    %sum32 = arith.trunci %sum1 : i64 to i32
    return %sum32 : i32
  }

  hw.module @ptrlen_abi_test_4args() {
    %fmtPrefix = sim.fmt.literal "result="
    %fmtNl = sim.fmt.literal "\0A"
    llhd.process {
      %zero64 = arith.constant 0 : i64
      %self = llvm.inttoptr %zero64 : i64 to !llvm.ptr
      %verb = arith.constant 10 : i32
      %sev = arith.constant 2 : i2
      %len = llvm.mlir.constant(123 : i64) : i64
      %id0 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
      %id1 = llvm.insertvalue %self, %id0[0] : !llvm.struct<(ptr, i64)>
      %id2 = llvm.insertvalue %len, %id1[1] : !llvm.struct<(ptr, i64)>
      %r = func.call @report_ptrlen_abi_func4(
          %self, %verb, %sev, %id2) :
          (!llvm.ptr, i32, i2, !llvm.struct<(ptr, i64)>) -> i32
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
// CHECK: result=135
