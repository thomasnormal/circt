// RUN: circt-sim %s --max-time=1000000000 2>&1 | FileCheck %s
//
// Regression: when simulating split hdl_top/hvl_top, values computed by
// module-level LLVM ops in hdl_top must remain available after hvl_top init.
// If they are lost, getValue may re-execute llvm.call on demand and diverge
// from already-initialized signal state.
//
// CHECK: [circt-sim] Simulating 2 top modules: hdl_top, hvl_top
// CHECK: module_init_preserved
// CHECK-NOT: module_init_recomputed

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  hw.module @hdl_top() {
    %c8 = llvm.mlir.constant(8 : i64) : i64
    %ptr = llvm.call @malloc(%c8) : (i64) -> !llvm.ptr
    %sig = llhd.sig %ptr : !llvm.ptr

    %fmt_ok = sim.fmt.literal "module_init_preserved\n"
    %fmt_bad = sim.fmt.literal "module_init_recomputed\n"

    llhd.process {
      %sigPtr = llhd.prb %sig : !llvm.ptr
      %ptrInt = llvm.ptrtoint %ptr : !llvm.ptr to i64
      %sigInt = llvm.ptrtoint %sigPtr : !llvm.ptr to i64
      %eq = llvm.icmp "eq" %ptrInt, %sigInt : i64
      cf.cond_br %eq, ^ok, ^bad
    ^ok:
      sim.proc.print %fmt_ok
      sim.terminate success, quiet
      llhd.halt
    ^bad:
      sim.proc.print %fmt_bad
      sim.terminate failure, quiet
      llhd.halt
    }

    hw.output
  }

  hw.module @hvl_top() {
    llhd.process {
      llhd.halt
    }
    hw.output
  }
}
