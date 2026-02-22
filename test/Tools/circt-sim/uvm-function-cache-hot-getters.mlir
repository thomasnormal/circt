// RUN: env CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 circt-sim %s 2>&1 | FileCheck %s
//
// Verify that hot UVM getter names (e.g. get_common_domain) are memoized.
// The function increments a counter on execution; with caching, only the first
// call executes the body and later calls hit the function-result cache.
//
// CHECK: calls=1

module {
  llvm.mlir.global internal @g_calls(0 : i32) : i32

  func.func private @get_common_domain() -> !llvm.ptr {
    %g = llvm.mlir.addressof @g_calls : !llvm.ptr
    %old = llvm.load %g : !llvm.ptr -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %old, %one : i32
    llvm.store %next, %g : i32, !llvm.ptr
    %addr = arith.constant 4096 : i64
    %ptr = llvm.inttoptr %addr : i64 to !llvm.ptr
    return %ptr : !llvm.ptr
  }

  hw.module @top() {
    %prefix = sim.fmt.literal "calls="
    %nl = sim.fmt.literal "\0A"

    llhd.process {
      %v0 = func.call @get_common_domain() : () -> !llvm.ptr
      %v1 = func.call @get_common_domain() : () -> !llvm.ptr
      %v2 = func.call @get_common_domain() : () -> !llvm.ptr

      %g = llvm.mlir.addressof @g_calls : !llvm.ptr
      %calls = llvm.load %g : !llvm.ptr -> i32
      %count = sim.fmt.dec %calls signed : i32
      %line = sim.fmt.concat (%prefix, %count, %nl)
      sim.proc.print %line
      llhd.halt
    }

    hw.output
  }
}
