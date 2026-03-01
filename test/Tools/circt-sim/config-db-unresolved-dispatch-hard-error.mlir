// RUN: not circt-sim %s --top test 2>&1 | FileCheck %s
//
// Regression: unresolved config_db call_indirect dispatch must fail loudly
// instead of silently falling back to heuristic interception.
//
// CHECK: CIRCTSIM-CFGDB-UNRESOLVED-DISPATCH
// CHECK-NOT: AFTER

module {
  hw.module @test() {
    %null = llvm.mlir.zero : !llvm.ptr
    %undef_str = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %c0_i64 = llvm.mlir.constant(0 : i64) : i64
    %c1_i64 = llvm.mlir.constant(1 : i64) : i64

    %empty_s0 = llvm.insertvalue %null, %undef_str[0] : !llvm.struct<(ptr, i64)>
    %empty_str = llvm.insertvalue %c0_i64, %empty_s0[1] : !llvm.struct<(ptr, i64)>

    %lit_after = sim.fmt.literal "AFTER"

    llhd.process {
      cf.br ^start
    ^start:
      %slot = llvm.alloca %c1_i64 x !llvm.ptr : (i64) -> !llvm.ptr
      llvm.store %null, %slot : !llvm.ptr, !llvm.ptr
      %slot_ref = builtin.unrealized_conversion_cast %slot : !llvm.ptr to !llhd.ref<!llvm.ptr>

      %fp = builtin.unrealized_conversion_cast %null : !llvm.ptr to (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llhd.ref<!llvm.ptr>) -> i1
      %ok = func.call_indirect %fp(%null, %null, %empty_str, %empty_str, %slot_ref) : (!llvm.ptr, !llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, !llhd.ref<!llvm.ptr>) -> i1

      %fmt_ok = sim.fmt.dec %ok : i1
      sim.proc.print %fmt_ok
      sim.proc.print %lit_after
      llhd.halt
    }
    hw.output
  }
}
