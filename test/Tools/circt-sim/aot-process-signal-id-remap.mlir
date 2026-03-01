// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --compiled=%t.so --aot-stats 2>&1 \
// RUN:   | FileCheck %s --check-prefix=COMPILED
// RUN: env CIRCT_AOT_DISABLE_ALL=1 \
// RUN:   circt-sim %s --compiled=%t.so --aot-stats 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DISABLE-ALL
//
// Regression: circt-compile assigns process signal IDs by root-module
// llhd.sig walk order, while runtime IDs are elaboration/instance order.
// If IDs are not remapped at runtime, the compiled callback toggles a wrong
// signal and clk prints become 0,0,0 instead of 0,1,0.
//
// COMPILE: [circt-compile] Compiled {{[1-9][0-9]*}} process bodies
//
// COMPILED: Compiled callback invocations: {{[1-9][0-9]*}}
// COMPILED: clk=0
// COMPILED: clk=1
// COMPILED: clk=0
//
// DISABLE-ALL: Compiled callback invocations: {{[[:space:]]*}}0
// DISABLE-ALL: clk=0
// DISABLE-ALL: clk=1
// DISABLE-ALL: clk=0

hw.module private @child() {
  %false = hw.constant false
  %c2_i64 = hw.constant 2000000 : i64
  %fmt_head = sim.fmt.literal "child="
  %fmt_nl = sim.fmt.literal "\0A"
  %sig = llhd.sig name "child_sig" %false : i1
  llhd.process {
    %d = llhd.int_to_time %c2_i64
    llhd.wait delay %d, ^bb1
  ^bb1:
    %v = llhd.prb %sig : i1
    %f = sim.fmt.dec %v : i1
    %o = sim.fmt.concat (%fmt_head, %f, %fmt_nl)
    sim.proc.print %o
    llhd.halt
  }
  hw.output
}

hw.module @top() {
  %false = hw.constant false
  %true  = hw.constant true
  %c1_i64  = hw.constant  1000000 : i64
  %c5_i64  = hw.constant  5000000 : i64
  %c25_i64 = hw.constant 25000000 : i64

  %fmt_clk = sim.fmt.literal "clk="
  %fmt_nl  = sim.fmt.literal "\0A"

  // Keep child instance ahead of top-level clk process to perturb compile-time
  // signal numbering versus runtime elaboration numbering.
  hw.instance "u" @child() -> ()

  %clk = llhd.sig name "clk" %false : i1

  // Time-only callback candidate that toggles clk every 5ns.
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
