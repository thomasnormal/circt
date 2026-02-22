// RUN: circt-sim %s --top top --max-time=1000000 | FileCheck %s

// Regression: When a signal has no llhd.drv users and is written via
// llvm.store through an unrealized_conversion_cast, a second store through a
// block argument must still be visible to llhd.prb on that signal.
//
// The bug: the first llvm.store (via direct cast result) sets
// pendingEpsilonDrives[sigId]. The second llvm.store goes through a block
// argument, so resolveSignalId fails and only writes to backing memory.
// llhd.prb then returns the stale pendingEpsilonDrives value.
//
// Fix: interpretLLVMStore syncs backing memory writes back to
// pendingEpsilonDrives and scheduler.updateSignal.

// CHECK: BEFORE=0
// CHECK: AFTER=1

module {
  hw.module @top() {
    %c1_i64 = llvm.mlir.constant(1 : i64) : i64

    %fmt_before = sim.fmt.literal "BEFORE="
    %fmt_after = sim.fmt.literal "AFTER="
    %fmt_nl = sim.fmt.literal "\0A"

    // Create a signal with NO llhd.drv users — only written via llvm.store.
    %init = hw.constant 0 : i1
    %sig = llhd.sig %init : i1

    llhd.process {
      // Get a pointer to the signal's backing memory.
      %ptr = builtin.unrealized_conversion_cast %sig : !llhd.ref<i1> to !llvm.ptr

      // First store: write 0 (initial). This goes through the direct cast
      // result, so resolveSignalId succeeds and sets pendingEpsilonDrives.
      %false = llvm.mlir.constant(false) : i1
      llvm.store %false, %ptr : i1, !llvm.ptr

      // Probe the signal — should be 0.
      %val0 = llhd.prb %sig : i1
      %fmt0 = sim.fmt.bin %val0 : i1
      %out0 = sim.fmt.concat (%fmt_before, %fmt0, %fmt_nl)
      sim.proc.print %out0

      // Branch to a new block, threading the pointer as a block argument.
      // This simulates what MooreToCore generates for conditional paths.
      cf.br ^write(%ptr : !llvm.ptr)

    ^write(%arg_ptr : !llvm.ptr):
      // Second store: write 1 through the block argument.
      // resolveSignalId will fail on %arg_ptr, so only backing memory updates.
      // Without the fix, pendingEpsilonDrives still has the old value (0).
      %true = llvm.mlir.constant(true) : i1
      llvm.store %true, %arg_ptr : i1, !llvm.ptr

      // Probe the signal — must be 1 (not stale 0).
      %val1 = llhd.prb %sig : i1
      %fmt1 = sim.fmt.bin %val1 : i1
      %out1 = sim.fmt.concat (%fmt_after, %fmt1, %fmt_nl)
      sim.proc.print %out1

      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
