// RUN: circt-sim %s | FileCheck %s

// CHECK: v=0

hw.module @test() {
  %false = hw.constant false
  %true = hw.constant true
  %c1_i64 = hw.constant 1000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_v = sim.fmt.literal "v="
  %fmt_nl = sim.fmt.literal "\0A"

  %s0 = hw.aggregate_constant [false] : !hw.struct<value: i1>
  %s1 = hw.aggregate_constant [true] : !hw.struct<value: i1>

  %sel = llhd.sig %false : i1
  %a = llhd.sig %s1 : !hw.struct<value: i1>
  %b = llhd.sig %s0 : !hw.struct<value: i1>
  %out = llhd.sig %s0 : !hw.struct<value: i1>

  %comb:1 = llhd.combinational -> !hw.struct<value: i1> {
    %sel_v = llhd.prb %sel : i1
    %ref = comb.mux %sel_v, %a, %b : !llhd.ref<!hw.struct<value: i1>>
    %val = llhd.prb %ref : !hw.struct<value: i1>
    llhd.yield %val : !hw.struct<value: i1>
  }
  llhd.drv %out, %comb#0 after %eps : !hw.struct<value: i1>

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %out_val = llhd.prb %out : !hw.struct<value: i1>
    %value = hw.struct_extract %out_val["value"] : !hw.struct<value: i1>
    %fmt_val = sim.fmt.dec %value : i1
    %fmt_out = sim.fmt.concat (%fmt_v, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    llhd.halt
  }

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
