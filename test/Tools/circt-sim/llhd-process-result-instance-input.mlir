// RUN: circt-sim %s | FileCheck %s

// CHECK: proc_in=1

hw.module private @child(in %in : i1, out out : i1) {
  %false = hw.constant false
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %sig = llhd.sig %false : i1
  llhd.drv %sig, %in after %eps : i1
  %val = llhd.prb %sig : i1
  hw.output %val : i1
}

hw.module @test() {
  %true = hw.constant true
  %c1_i64 = hw.constant 1000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_pre = sim.fmt.literal "proc_in="
  %fmt_nl = sim.fmt.literal "\0A"

  %proc_val = llhd.process -> i1 {
    llhd.wait yield (%true : i1), delay %eps, ^bb1
  ^bb1:
    llhd.halt %true : i1
  }

  %inst.out = hw.instance "u" @child(in: %proc_val : i1) -> (out: i1)

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %fmt_val = sim.fmt.dec %inst.out : i1
    %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
