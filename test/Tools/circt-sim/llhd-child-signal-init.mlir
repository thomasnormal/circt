// RUN: circt-sim %s | FileCheck %s

// CHECK: child_init=1

hw.module private @child(out out : i1) {
  %true = hw.constant true
  %sig = llhd.sig %true : i1
  %val = llhd.prb %sig : i1
  hw.output %val : i1
}

hw.module @test() {
  %c1_i64 = hw.constant 1000000 : i64
  %fmt_pre = sim.fmt.literal "child_init="
  %fmt_nl = sim.fmt.literal "\0A"

  %inst.out = hw.instance "u" @child() -> (out: i1)

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %fmt_val = sim.fmt.bin %inst.out : i1
    %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
