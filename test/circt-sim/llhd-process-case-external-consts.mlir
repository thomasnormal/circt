// RUN: circt-sim %s --top=test_case_external_consts 2>&1 | FileCheck %s
// RUN: env CIRCT_SIM_ENABLE_DIRECT_FASTPATHS=1 circt-sim %s --top=test_case_external_consts 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Regression test: case-style cf.cond_br chains that compare against
// module-scope hw.constant values must evaluate all arms correctly.
//
// CHECK: sel=0 y=170
// CHECK: sel=1 y=187
// CHECK: sel=2 y=204
// CHECK: sel=3 y=221

hw.module @test_case_external_consts() {
  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  %c2_i2 = hw.constant 2 : i2
  %c3_i2 = hw.constant 3 : i2

  %c170_i32 = hw.constant 170 : i32
  %c187_i32 = hw.constant 187 : i32
  %c204_i32 = hw.constant 204 : i32
  %c221_i32 = hw.constant 221 : i32

  %fmt_nl = sim.fmt.literal "\0A"
  %fmt_s0 = sim.fmt.literal "sel=0 y="
  %fmt_s1 = sim.fmt.literal "sel=1 y="
  %fmt_s2 = sim.fmt.literal "sel=2 y="
  %fmt_s3 = sim.fmt.literal "sel=3 y="

  %t0 = llhd.constant_time <0ns, 0d, 0e>
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %t1 = llhd.constant_time <1ns, 0d, 0e>

  %sel = llhd.sig %c0_i2 : i2
  %y = llhd.sig %c170_i32 : i32

  // Case-like lowering pattern with a default arm.
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %s = llhd.prb %sel : i2
    %eq0 = comb.icmp eq %s, %c0_i2 : i2
    cf.cond_br %eq0, ^bb2, ^bb3
  ^bb2:
    llhd.drv %y, %c170_i32 after %eps : i32
    cf.br ^bb8
  ^bb3:
    %eq1 = comb.icmp eq %s, %c1_i2 : i2
    cf.cond_br %eq1, ^bb4, ^bb5
  ^bb4:
    llhd.drv %y, %c187_i32 after %eps : i32
    cf.br ^bb8
  ^bb5:
    %eq2 = comb.icmp eq %s, %c2_i2 : i2
    cf.cond_br %eq2, ^bb6, ^bb7
  ^bb6:
    llhd.drv %y, %c204_i32 after %eps : i32
    cf.br ^bb8
  ^bb7:
    llhd.drv %y, %c221_i32 after %eps : i32
    cf.br ^bb8
  ^bb8:
    %s_obs = llhd.prb %sel : i2
    llhd.wait (%s_obs : i2), ^bb1
  }

  llhd.process {
    llhd.drv %sel, %c0_i2 after %t0 : i2
    llhd.wait delay %t1, ^p1
  ^p1:
    %y0 = llhd.prb %y : i32
    %f0v = sim.fmt.dec %y0 : i32
    %f0 = sim.fmt.concat (%fmt_s0, %f0v, %fmt_nl)
    sim.proc.print %f0

    llhd.drv %sel, %c1_i2 after %t0 : i2
    llhd.wait delay %t1, ^p2
  ^p2:
    %y1 = llhd.prb %y : i32
    %f1v = sim.fmt.dec %y1 : i32
    %f1 = sim.fmt.concat (%fmt_s1, %f1v, %fmt_nl)
    sim.proc.print %f1

    llhd.drv %sel, %c2_i2 after %t0 : i2
    llhd.wait delay %t1, ^p3
  ^p3:
    %y2 = llhd.prb %y : i32
    %f2v = sim.fmt.dec %y2 : i32
    %f2 = sim.fmt.concat (%fmt_s2, %f2v, %fmt_nl)
    sim.proc.print %f2

    llhd.drv %sel, %c3_i2 after %t0 : i2
    llhd.wait delay %t1, ^p4
  ^p4:
    %y3 = llhd.prb %y : i32
    %f3v = sim.fmt.dec %y3 : i32
    %f3 = sim.fmt.concat (%fmt_s3, %f3v, %fmt_nl)
    sim.proc.print %f3

    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
