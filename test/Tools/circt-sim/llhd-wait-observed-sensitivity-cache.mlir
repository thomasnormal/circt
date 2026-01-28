// RUN: circt-sim --sim-stats --process-stats %s | FileCheck %s

// CHECK: === Process Stats
// CHECK: steps=

hw.module @test() {
  %c1_i64 = hw.constant 1000000 : i64
  %c2_i64 = hw.constant 2000000 : i64
  %c3_i64 = hw.constant 3000000 : i64
  %true = hw.constant true
  %false = hw.constant false
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %a = llhd.sig %false : i1
  %out = llhd.sig %false : i1

  llhd.process {
    %d1 = llhd.int_to_time %c1_i64
    %d2 = llhd.int_to_time %c2_i64
    %d3 = llhd.int_to_time %c3_i64
    llhd.wait delay %d1, ^bb1
  ^bb1:
    llhd.drv %a, %true after %eps : i1
    llhd.wait delay %d2, ^bb2
  ^bb2:
    llhd.drv %a, %false after %eps : i1
    llhd.wait delay %d3, ^bb3
  ^bb3:
    sim.terminate success, quiet
    llhd.halt
  }

  llhd.process {
    %a_val = llhd.prb %a : i1
    llhd.drv %out, %a_val after %eps : i1
    llhd.wait (%a_val : i1), ^bb1
  ^bb1:
    %a_val1 = llhd.prb %a : i1
    llhd.drv %out, %a_val1 after %eps : i1
    llhd.wait (%a_val1 : i1), ^bb1
  }

  hw.output
}
