// RUN: circt-compile %s -o %t.so
// RUN: circt-sim %s --top cache_stale_ctor_test --mode=compile --compiled=%t.so --aot-stats 2>&1 | FileCheck %s

// Regression: func.call cache entries can be created during module-init
// global_ctors before AOT native function tables are loaded. The same callsite
// may run again after AOT load; make sure cached entries get native backfill.
// Expected fallback count for @inc_counter is exactly one (the pre-load call
// from global ctor init), not two.

module {
  llvm.mlir.global internal @g_counter(0 : i64) {addr_space = 0 : i32} : i64

  func.func private @inc_counter() -> i64 {
    %g = llvm.mlir.addressof @g_counter : !llvm.ptr
    %old = llvm.load %g : !llvm.ptr -> i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %new = arith.addi %old, %c1 : i64
    llvm.store %new, %g : i64, !llvm.ptr
    return %new : i64
  }

  llvm.func internal @__ctor_twice() {
    %v = func.call @inc_counter() : () -> i64
    llvm.return
  }

  llvm.mlir.global_ctors ctors = [@__ctor_twice], priorities = [65535 : i32], data = [#llvm.zero]

  hw.module @cache_stale_ctor_test() {
    %fmt_prefix = sim.fmt.literal "counter="
    %fmt_nl = sim.fmt.literal "\0A"
    llhd.process {
      // Re-execute the same ctor function after AOT tables are loaded.
      // The nested func.call @inc_counter should be native-capable now.
      llvm.call @__ctor_twice() : () -> ()
      %g = llvm.mlir.addressof @g_counter : !llvm.ptr
      %val = llvm.load %g : !llvm.ptr -> i64
      %fmt_val = sim.fmt.dec %val specifierWidth 0 : i64
      %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
      sim.proc.print %fmt_out
      llhd.halt
    }
    hw.output
  }
}

// CHECK: Top interpreted func.call fallback reasons
// CHECK: inc_counter [no-native=1]
// CHECK: counter=2
