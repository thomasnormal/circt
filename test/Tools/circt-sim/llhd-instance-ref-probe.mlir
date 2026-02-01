// RUN: circt-sim %s | FileCheck %s

// CHECK: ref_probe=0

hw.module private @child(out out : !llhd.ref<i1>) {
  %false = hw.constant false
  %sig = llhd.sig %false : i1
  hw.output %sig : !llhd.ref<i1>
}

hw.module @test() {
  %c1_i64 = hw.constant 1000000 : i64
  %fmt_pre = sim.fmt.literal "ref_probe="
  %fmt_nl = sim.fmt.literal "\0A"

  %inst.out = hw.instance "u" @child() -> (out: !llhd.ref<i1>)

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %val = llhd.prb %inst.out : i1
    %fmt_val = sim.fmt.bin %val : i1
    %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
