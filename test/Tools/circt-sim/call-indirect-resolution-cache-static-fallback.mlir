// RUN: env CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 circt-sim %s 2>&1 | FileCheck %s
//
// Test static fallback resolution via vtable entries attribute.
//
// CHECK: ok
// CHECK: [circt-sim] call_indirect resolution cache: installs=

module {
  func.func @static_method() {
    return
  }

  llvm.mlir.global internal @vtable(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = [[0, @static_method]]} : !llvm.array<1 x ptr>

  hw.module @top() {
    %fmt_ok = sim.fmt.literal "ok\0A"

    llhd.process {
      cf.br ^entry
    ^entry:
      %vt = llvm.mlir.addressof @vtable : !llvm.ptr
      %c0 = arith.constant 0 : i64

      %slot0 = llvm.getelementptr %vt[%c0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fp0 = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
      %fp0_cast = builtin.unrealized_conversion_cast %fp0 : !llvm.ptr to () -> ()

      func.call_indirect %fp0_cast() : () -> ()

      sim.proc.print %fmt_ok
      llhd.halt
    }
    hw.output
  }
}
