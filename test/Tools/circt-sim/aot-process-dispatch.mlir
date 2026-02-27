// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so --aot-stats 2>&1 | FileCheck %s --check-prefix=STATS
//
// Test that an eligible single-fire driver process is dispatched through the
// compiled AOT callback path rather than the interpreter.
//
// The driver process (1 wait, drives 1-bit signal) is eligible for AOT
// compilation. After Phase L, running with --compiled should route that
// process through the compiled entry point, giving non-zero compiled callback
// invocations while producing identical output.
//
// COMPILE: [circt-compile] Compiled 1 process bodies
//
// SIM: b=1
// SIM: b=0
//
// STATS: Compiled callback invocations: {{[1-9][0-9]*}}
// STATS: b=1
// STATS: b=0

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

  // Driver: set a=true at t=5ns. 1 wait, ≤64-bit drive → eligible for AOT.
  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^set
  ^set:
    %eps2 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %a, %true after %eps2 : i1
    llhd.halt
  }

  // Reader: print b at t=1ns, then t=7ns.
  llhd.process {
    %d0 = llhd.int_to_time %c1_i64
    llhd.wait delay %d0, ^r1
  ^r1:
    %b1  = llhd.prb %b : i1
    %v1  = sim.fmt.dec %b1 : i1
    %o1  = sim.fmt.concat (%fmt_b, %v1, %fmt_nl)
    sim.proc.print %o1
    %d1 = llhd.int_to_time %c6_i64
    llhd.wait delay %d1, ^r2
  ^r2:
    %b2  = llhd.prb %b : i1
    %v2  = sim.fmt.dec %b2 : i1
    %o2  = sim.fmt.concat (%fmt_b, %v2, %fmt_nl)
    sim.proc.print %o2
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
