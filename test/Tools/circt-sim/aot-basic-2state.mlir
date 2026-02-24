// RUN: circt-sim %s | FileCheck %s
// RUN: circt-sim-compile %s -o %t.so -v 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --max-time=20000000 --compiled=%t.so --aot-stats 2>&1 | FileCheck %s --check-prefix=DISPATCH

// Test AOT process body compilation for a 2-state signal inverter.
// The driver process (1 wait, drives 1-bit signal) is eligible for AOT
// compilation as a callback function. The simulation output must be identical
// whether running interpreted or with the compiled .so.
//
// Simulation: b = NOT a.  a starts false, becomes true at t=5ns.
//   t=1ns:  b=1  (NOT false)
//   t=7ns:  b=0  (NOT true, after a toggled at t=5ns)
//
// CHECK: b=1
// CHECK: b=0
//
// COMPILE: [circt-sim-compile] Compiled 1 process bodies
// COMPILE: [circt-sim-compile] 0 functions + 1 processes ready for codegen
//
// DISPATCH: Compiled process dispatch: 1/1 processes wired

hw.module @test() {
  %false = hw.constant false
  %true  = hw.constant true
  %c1_i64  = hw.constant  1000000 : i64
  %c5_i64  = hw.constant  5000000 : i64
  %c6_i64  = hw.constant  6000000 : i64
  %c20_i64 = hw.constant 20000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_b  = sim.fmt.literal "b="
  %fmt_nl = sim.fmt.literal "\0A"

  // Signals: a (input), b (output = NOT a).
  %a = llhd.sig %false : i1
  %b = llhd.sig %false : i1

  // Combinational: b = NOT a.
  %comb:1 = llhd.combinational -> i1 {
    %a_val = llhd.prb %a : i1
    %not_a = comb.xor %a_val, %true : i1
    llhd.yield %not_a : i1
  }
  llhd.drv %b, %comb#0 after %eps : i1

  // Driver process: set a=true at t=5ns.
  // Single wait + integer drive ≤64 bits → eligible for AOT compilation.
  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^set
  ^set:
    %eps2 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %a, %true after %eps2 : i1
    llhd.halt
  }

  // Reader: print b at t=1ns, then t=7ns (1ns + 6ns).
  llhd.process {
    %d0 = llhd.int_to_time %c1_i64
    llhd.wait delay %d0, ^read1
  ^read1:
    %b1    = llhd.prb %b : i1
    %v1    = sim.fmt.dec %b1 : i1
    %out1  = sim.fmt.concat (%fmt_b, %v1, %fmt_nl)
    sim.proc.print %out1
    %d1 = llhd.int_to_time %c6_i64
    llhd.wait delay %d1, ^read2
  ^read2:
    %b2    = llhd.prb %b : i1
    %v2    = sim.fmt.dec %b2 : i1
    %out2  = sim.fmt.concat (%fmt_b, %v2, %fmt_nl)
    sim.proc.print %out2
    llhd.halt
  }

  // Terminator at t=20ns.
  llhd.process {
    %d = llhd.int_to_time %c20_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
