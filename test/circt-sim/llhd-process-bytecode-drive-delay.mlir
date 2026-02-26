// RUN: circt-sim %s --top=test_bytecode_drive_delay 2>&1 | FileCheck %s
// RUN: env CIRCT_SIM_ENABLE_DIRECT_FASTPATHS=1 circt-sim %s --top=test_bytecode_drive_delay 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Regression test: direct fast-path bytecode execution must preserve llhd.drv
// delay semantics. A non-trivial drive delay must not be collapsed to same-slot
// update timing.
//
// CHECK: early=0
// CHECK: late=1

hw.module @test_bytecode_drive_delay() {
  %c0_i1 = hw.constant false
  %c1_i1 = hw.constant true
  %t0 = llhd.constant_time <0ns, 0d, 0e>
  %t1 = llhd.constant_time <1ns, 0d, 0e>
  %t10 = llhd.constant_time <10ns, 0d, 0e>

  %fmt_nl = sim.fmt.literal "\0A"
  %fmt_early = sim.fmt.literal "early="
  %fmt_late = sim.fmt.literal "late="

  %in = llhd.sig %c0_i1 : i1
  %out = llhd.sig %c0_i1 : i1

  // Wait on input edges and forward to output with a real-time delay.
  llhd.process {
    cf.br ^wait
  ^wait:
    %observed = llhd.prb %in : i1
    llhd.wait (%observed : i1), ^active
  ^active:
    %v = llhd.prb %in : i1
    llhd.drv %out, %v after %t10 : i1
    cf.br ^wait
  }

  llhd.process {
    llhd.drv %in, %c1_i1 after %t0 : i1
    llhd.wait delay %t1, ^check_early
  ^check_early:
    %oe = llhd.prb %out : i1
    %fev = sim.fmt.dec %oe : i1
    %fe = sim.fmt.concat (%fmt_early, %fev, %fmt_nl)
    sim.proc.print %fe

    llhd.wait delay %t10, ^check_late
  ^check_late:
    %ol = llhd.prb %out : i1
    %flv = sim.fmt.dec %ol : i1
    %fl = sim.fmt.concat (%fmt_late, %flv, %fmt_nl)
    sim.proc.print %fl

    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
