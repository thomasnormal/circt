// RUN: circt-sim %s | FileCheck %s

// CHECK: inst_out=1

hw.module private @child(in %in : i1, out out : i1) {
  hw.output %in : i1
}

hw.module @test() {
  %false = hw.constant false
  %true = hw.constant true
  %c1_i64 = hw.constant 1000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_pre = sim.fmt.literal "inst_out="
  %fmt_nl = sim.fmt.literal "\0A"

  %sig_in = llhd.sig %false : i1
  %sig_in_val = llhd.prb %sig_in : i1
  %inst.out = hw.instance "u" @child(in: %sig_in_val : i1) -> (out: i1)

  llhd.process {
    llhd.drv %sig_in, %false after %eps : i1
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.drv %sig_in, %true after %eps : i1
    llhd.halt
  }

  llhd.process {
    llhd.wait (%inst.out : i1), ^bb1
  ^bb1:
    %fmt_val = sim.fmt.dec %inst.out : i1
    %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
