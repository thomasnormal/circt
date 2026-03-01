// RUN: circt-compile %s -o %t.so
// RUN: circt-sim %s --top ptrlen_abi_test_2args_method --mode=compile --compiled=%t.so --aot-stats 2>&1 | FileCheck %s

// Regression/perf guard: allow direct native func.call for method-like
// two-arg ABI shape (!llvm.ptr, !llvm.struct<(ptr, i64)>) for internal
// compiled callees.

module {
  func.func private @internal_ptrlen_method(
      %self: !llvm.ptr, %name: !llvm.struct<(ptr, i64)>) -> !llvm.ptr {
    return %self : !llvm.ptr
  }

  hw.module @ptrlen_abi_test_2args_method() {
    %fmtPrefix = sim.fmt.literal "result="
    %fmtNl = sim.fmt.literal "\0A"
    llhd.process {
      %one = llvm.mlir.constant(1 : i64) : i64
      %self = llvm.alloca %one x i8 : (i64) -> !llvm.ptr
      %len = llvm.mlir.constant(7 : i64) : i64
      %name0 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
      %name1 = llvm.insertvalue %self, %name0[0] : !llvm.struct<(ptr, i64)>
      %name2 = llvm.insertvalue %len, %name1[1] : !llvm.struct<(ptr, i64)>
      %r = func.call @internal_ptrlen_method(%self, %name2) :
          (!llvm.ptr, !llvm.struct<(ptr, i64)>) -> !llvm.ptr
      %r64 = llvm.ptrtoint %r : !llvm.ptr to i64
      %zero = llvm.mlir.constant(0 : i64) : i64
      %isNonNull = arith.cmpi ne, %r64, %zero : i64
      %isNonNullI32 = arith.extui %isNonNull : i1 to i32
      %fmtDec = sim.fmt.dec %isNonNullI32 signed : i32
      %fmtOut = sim.fmt.concat (%fmtPrefix, %fmtDec, %fmtNl)
      sim.proc.print %fmtOut
      llhd.halt
    }
    hw.output
  }
}

// CHECK: direct_calls_native:              1
// CHECK: direct_calls_interpreted:         0
// CHECK: result=1
