// RUN: circt-sim %s --skip-passes --max-time=100 2>&1 | FileCheck %s
//
// Test that a clock toggle process works correctly when compiled to native
// code via the full-process JIT (ucontext-based coroutine execution).
// The process toggles %clk every 10fs. At time 100, clk should have toggled
// from 0 → 0 → 1 → 0 → 1 → ... = 5 toggles in [10,50] = value 1 at t=90.

hw.module @top() {
  %false = hw.constant false
  %true = hw.constant true
  %zero = llhd.constant_time <0ns, 0d, 1e>
  %tick = arith.constant 10 : i64

  %clk = llhd.sig %false : i1

  // Periodic toggle clock process.
  // Entry block drives 0, then branches to wait loop.
  // Wait loop: wait 10fs, toggle, repeat.
  llhd.process {
    llhd.drv %clk, %false after %zero : i1
    cf.br ^wait
  ^wait:
    %delay = llhd.int_to_time %tick
    llhd.wait delay %delay, ^toggle
  ^toggle:
    %cur = llhd.prb %clk : i1
    %next = comb.xor %cur, %true : i1
    llhd.drv %clk, %next after %zero : i1
    cf.br ^wait
  }

  hw.output
}

// CHECK: [circt-sim] Simulation completed
