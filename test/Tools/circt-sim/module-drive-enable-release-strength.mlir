// RUN: circt-sim %s | FileCheck %s

// Regression: strength-resolved continuous drives with an enable must release
// their per-drive contribution when the enable deasserts.
//
// CHECK: pre_v=0 pre_u=0
// CHECK: post_v=1 post_u=0

hw.module @test() {
  %c1_i64 = hw.constant 1000000 : i64
  %false = hw.constant false
  %true = hw.constant true
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_pre_v = sim.fmt.literal "pre_v="
  %fmt_pre_u = sim.fmt.literal " pre_u="
  %fmt_post_v = sim.fmt.literal "post_v="
  %fmt_post_u = sim.fmt.literal " post_u="
  %fmt_nl = sim.fmt.literal "\0A"

  %v0 = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
  %v1 = hw.aggregate_constant [true, false] : !hw.struct<value: i1, unknown: i1>
  %vz = hw.aggregate_constant [true, true] : !hw.struct<value: i1, unknown: i1>

  %en_sig = llhd.sig %true : i1
  %bus_sig = llhd.sig %vz : !hw.struct<value: i1, unknown: i1>

  llhd.drv %bus_sig, %v1 after %eps strength(highz, pull) : !hw.struct<value: i1, unknown: i1>
  %en_val = llhd.prb %en_sig : i1
  llhd.drv %bus_sig, %v0 after %eps if %en_val : !hw.struct<value: i1, unknown: i1>

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %pre = llhd.prb %bus_sig : !hw.struct<value: i1, unknown: i1>
    %pre_v = hw.struct_extract %pre["value"] : !hw.struct<value: i1, unknown: i1>
    %pre_u = hw.struct_extract %pre["unknown"] : !hw.struct<value: i1, unknown: i1>
    %fmt_pre_vv = sim.fmt.bin %pre_v : i1
    %fmt_pre_uv = sim.fmt.bin %pre_u : i1
    %fmt_pre = sim.fmt.concat (%fmt_pre_v, %fmt_pre_vv, %fmt_pre_u, %fmt_pre_uv, %fmt_nl)
    sim.proc.print %fmt_pre

    llhd.drv %en_sig, %false after %eps : i1

    llhd.wait delay %delay, ^bb2
  ^bb2:
    %post = llhd.prb %bus_sig : !hw.struct<value: i1, unknown: i1>
    %post_v = hw.struct_extract %post["value"] : !hw.struct<value: i1, unknown: i1>
    %post_u = hw.struct_extract %post["unknown"] : !hw.struct<value: i1, unknown: i1>
    %fmt_post_vv = sim.fmt.bin %post_v : i1
    %fmt_post_uv = sim.fmt.bin %post_u : i1
    %fmt_post = sim.fmt.concat (%fmt_post_v, %fmt_post_vv, %fmt_post_u, %fmt_post_uv, %fmt_nl)
    sim.proc.print %fmt_post

    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
