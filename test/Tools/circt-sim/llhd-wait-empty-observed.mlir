// RUN: circt-sim %s | FileCheck %s

// CHECK: sig=1

hw.module @WaitEmptyObserved() {
  %false = hw.constant false
  %true = hw.constant true
  %c1_i64 = hw.constant 1000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_pre = sim.fmt.literal "sig="
  %fmt_nl = sim.fmt.literal "\0A"

  %sig = llhd.sig %false : i1

  // Drive sig after 1ns.
  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.drv %sig, %true after %eps : i1
    llhd.halt
  }

  // Wait with no observed list; should derive sensitivity from probe.
  llhd.process {
    llhd.wait ^bb1
  ^bb1:
    %val = llhd.prb %sig : i1
    %fmt_val = sim.fmt.dec %val : i1
    %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
