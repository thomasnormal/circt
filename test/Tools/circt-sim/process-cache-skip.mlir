// RUN: circt-sim --sim-stats --process-stats %s | FileCheck %s

// CHECK: === Process Stats
// CHECK: steps=

hw.module @test() {
  %true = hw.constant true
  %false = hw.constant false
  %sig = llhd.sig %false : i1
  %out = llhd.sig %false : i1
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %term_delay = llhd.constant_time <0ns, 0d, 3e>

  llhd.process {
    llhd.wait delay %eps, ^bb1
  ^bb1:
    llhd.drv %sig, %true after %eps : i1
    llhd.drv %sig, %false after %eps : i1
    llhd.halt
  }

  llhd.process {
    %val = llhd.prb %sig : i1
    llhd.drv %out, %val after %eps : i1
    llhd.wait (%val : i1), ^bb1
  ^bb1:
    %val1 = llhd.prb %sig : i1
    llhd.drv %out, %val1 after %eps : i1
    llhd.wait (%val1 : i1), ^bb1
  }

  llhd.process {
    llhd.wait delay %term_delay, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
