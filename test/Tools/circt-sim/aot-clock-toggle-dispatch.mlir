// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so --aot-stats 2>&1 | FileCheck %s --check-prefix=STATS
//
// Test that an eligible looping clock-toggle process is dispatched through the
// compiled AOT callback path on every activation.
//
// The clock toggle process (1 wait, loops via cf.br, drives 1-bit signal) is
// eligible for AOT compilation and activates multiple times per simulation.
// After Phase L, compiled callback invocations should be ≥ 4 (one per 5ns
// period in a 25ns window).
//
// Simulation timeline (clk starts false):
//   t=0ns:  clk=0 → process enters, waits 5ns
//   t=5ns:  toggle → clk=1
//   t=10ns: toggle → clk=0
//   t=15ns: toggle → clk=1
//   t=20ns: toggle → clk=0
//   t=1ns:  print clk=0
//   t=6ns:  print clk=1
//   t=11ns: print clk=0
//
// COMPILE: [circt-sim-compile] Compiled 1 process bodies
//
// SIM: clk=0
// SIM: clk=1
// SIM: clk=0
//
// STATS: Compiled callback invocations: {{[1-9][0-9]*}}
// STATS: clk=0
// STATS: clk=1
// STATS: clk=0

hw.module @test() {
  %false = hw.constant false
  %true  = hw.constant true
  %c1_i64  = hw.constant  1000000 : i64
  %c5_i64  = hw.constant  5000000 : i64
  %c25_i64 = hw.constant 25000000 : i64

  %fmt_clk = sim.fmt.literal "clk="
  %fmt_nl  = sim.fmt.literal "\0A"

  // Signal: clk starts low.
  %clk = llhd.sig %false : i1

  // Clock toggle process: flip clk every 5ns.
  // Pattern: entry → bb1 (wait 5ns) → bb2 (toggle+drive) → bb1 (loop).
  // 1 wait, ≤64-bit probe/drive → eligible for AOT compilation.
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %half = llhd.int_to_time %c5_i64
    llhd.wait delay %half, ^bb2
  ^bb2:
    %v    = llhd.prb %clk : i1
    %n    = comb.xor %v, %true : i1
    %eps  = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %clk, %n after %eps : i1
    cf.br ^bb1
  }

  // Reader: print clk at t=1ns, t=6ns, t=11ns.
  llhd.process {
    %d0 = llhd.int_to_time %c1_i64
    llhd.wait delay %d0, ^r1
  ^r1:
    %v1  = llhd.prb %clk : i1
    %f1  = sim.fmt.dec %v1 : i1
    %o1  = sim.fmt.concat (%fmt_clk, %f1, %fmt_nl)
    sim.proc.print %o1
    %d1 = llhd.int_to_time %c5_i64
    llhd.wait delay %d1, ^r2
  ^r2:
    %v2  = llhd.prb %clk : i1
    %f2  = sim.fmt.dec %v2 : i1
    %o2  = sim.fmt.concat (%fmt_clk, %f2, %fmt_nl)
    sim.proc.print %o2
    %d2 = llhd.int_to_time %c5_i64
    llhd.wait delay %d2, ^r3
  ^r3:
    %v3  = llhd.prb %clk : i1
    %f3  = sim.fmt.dec %v3 : i1
    %o3  = sim.fmt.concat (%fmt_clk, %f3, %fmt_nl)
    sim.proc.print %o3
    llhd.halt
  }

  // Terminator at t=25ns.
  llhd.process {
    %d = llhd.int_to_time %c25_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
