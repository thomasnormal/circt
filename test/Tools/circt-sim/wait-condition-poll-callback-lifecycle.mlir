// RUN: env CIRCT_SIM_TRACE_WAIT_CONDITION_POLL_CALLBACK=1 circt-sim %s --max-time=1000 2>&1 | FileCheck %s

// CHECK: [WAITCOND-CB] install
// CHECK: [WAITCOND-CB] poll#1
// CHECK: [WAITCOND-CB] clear polls=1
// CHECK: done

module {
  func.func private @__moore_wait_condition(i32)

  hw.module @top() {
    %c0 = arith.constant 0 : i32
    %fmt = sim.fmt.literal "done\0A"

    llhd.process {
      // Use func.call (not llvm.call) so the call resolves through the
      // runtime symbol path and exercises the poll callback bridge.
      func.call @__moore_wait_condition(%c0) : (i32) -> ()
      sim.proc.print %fmt
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
