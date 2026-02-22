// RUN: circt-sim --skip-passes %s 2>&1 | FileCheck %s

// Test that child modules can correctly resolve parent signal values through
// block argument mappings when the signal was initialized by a module-level
// LLVM op (e.g., malloc).
//
// This covers a fix where llhd.prb results were cached as X during parent
// module op processing (because ProbeOp was evaluated on-demand before signal
// values were updated). The stale X was then propagated via moduleInitValueMap,
// causing child modules to resolve parent interface pointers as 0.
//
// The fix re-evaluates stale ProbeOp X entries after signal values are updated,
// ensuring moduleInitValueMap contains the correct values for child modules.
//
// Expected: child_ptr should be non-zero (not "0\n" or "x\n").
// Without the fix, child_ptr would be "0" due to stale X cache.

// CHECK: [circt-sim] Starting simulation
// CHECK-NOT: child_ptr=0
// CHECK-NOT: child_ptr=x
// CHECK: child_ptr=
// CHECK: [circt-sim] Simulation completed

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  // Child module: receives parent pointer and prints its address.
  // Without the ProbeOp cache fix, the block arg resolves to 0 (stale X).
  hw.module private @checker(in %ptr : !llvm.ptr) {
    %c1000000_i64 = hw.constant 1000000 : i64

    llhd.process {
      %delay = llhd.int_to_time %c1000000_i64
      llhd.wait delay %delay, ^bb1
    ^bb1:
      // Cast pointer to i64 to print its address
      %addr = llvm.ptrtoint %ptr : !llvm.ptr to i64
      // Truncate to i32 for fmt.dec
      %addr32 = llvm.trunc %addr : i64 to i32
      %fmt_pre = sim.fmt.literal "child_ptr="
      %fmt_nl = sim.fmt.literal "\0A"
      %fmt_val = sim.fmt.dec %addr32 : i32
      %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
      sim.proc.print %fmt_out
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }

  // Parent module: allocates memory, creates signal with pointer, probes it,
  // stores through it (triggering the ProbeOp cache bug), then passes the
  // probed pointer to a child instance.
  hw.module @test() {
    %c8_i64 = llvm.mlir.constant(8 : i64) : i64
    %c42_i32 = llvm.mlir.constant(42 : i32) : i32

    // Allocate memory (returns non-zero pointer)
    %ptr = llvm.call @malloc(%c8_i64) : (i64) -> !llvm.ptr

    // Create signal holding the pointer
    %sig = llhd.sig %ptr : !llvm.ptr

    // Probe the signal
    %probed = llhd.prb %sig : !llvm.ptr

    // GEP store through probed pointer — triggers on-demand ProbeOp
    // evaluation before signal update, caching stale X result
    %field0 = llvm.getelementptr %probed[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"interface.Bus", (i32, i32)>
    llvm.store %c42_i32, %field0 : i32, !llvm.ptr

    // Pass probed pointer to child
    hw.instance "child" @checker(ptr: %probed : !llvm.ptr) -> ()

    hw.output
  }
}
