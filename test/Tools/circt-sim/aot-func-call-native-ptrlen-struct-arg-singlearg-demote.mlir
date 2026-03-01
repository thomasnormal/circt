// RUN: circt-compile %s -o %t.so
// RUN: circt-sim %s --top ptrlen_singlearg_demote_test --mode=compile --compiled=%t.so --aot-stats 2>&1 | FileCheck %s

// Safety guard: do not broadly native-dispatch global one-argument
// ptr+len helpers. Keep these interpreted unless explicitly whitelisted.

module {
  func.func private @global_ptrlen_helper(
      %arg0: !llvm.struct<(ptr, i64)>) -> i32 {
    %len = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr, i64)>
    %len32 = arith.trunci %len : i64 to i32
    return %len32 : i32
  }

  hw.module @ptrlen_singlearg_demote_test() {
    %fmtPrefix = sim.fmt.literal "result="
    %fmtNl = sim.fmt.literal "\0A"
    llhd.process {
      %zero64 = arith.constant 0 : i64
      %self = llvm.inttoptr %zero64 : i64 to !llvm.ptr
      %len = llvm.mlir.constant(123 : i64) : i64
      %id0 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
      %id1 = llvm.insertvalue %self, %id0[0] : !llvm.struct<(ptr, i64)>
      %id2 = llvm.insertvalue %len, %id1[1] : !llvm.struct<(ptr, i64)>
      %r = func.call @global_ptrlen_helper(%id2) :
          (!llvm.struct<(ptr, i64)>) -> i32
      %fmtDec = sim.fmt.dec %r signed : i32
      %fmtOut = sim.fmt.concat (%fmtPrefix, %fmtDec, %fmtNl)
      sim.proc.print %fmtOut
      llhd.halt
    }
    hw.output
  }
}

// CHECK: direct_calls_native:              0
// CHECK: direct_calls_interpreted:         1
// CHECK: result=123
