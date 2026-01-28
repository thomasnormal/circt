// RUN: circt-sim --sim-stats --process-stats %s | FileCheck %s

// CHECK: === Process Stats
// CHECK: steps=

hw.module @test() {
  %c1_i64 = hw.constant 1000000 : i64
  %c2_i64 = hw.constant 2000000 : i64
  %true = hw.constant true
  %false = hw.constant false
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %a = llhd.sig %false : i1
  %out = llhd.sig %false : i1

  llhd.process {
    %d1 = llhd.int_to_time %c1_i64
    %d2 = llhd.int_to_time %c2_i64
    llhd.wait delay %d1, ^bb1
  ^bb1:
    llhd.drv %a, %true after %eps : i1
    llhd.wait delay %d2, ^bb2
  ^bb2:
    sim.terminate success, quiet
    llhd.halt
  }

  llhd.process {
    cf.br ^bb1
  ^bb1:
    %a_val = llhd.prb %a : i1
    llhd.drv %out, %a_val after %eps : i1
    llhd.wait ^bb1
  }

  hw.output
}
