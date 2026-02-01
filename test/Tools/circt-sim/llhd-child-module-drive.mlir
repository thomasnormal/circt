// RUN: circt-sim %s | FileCheck %s

// CHECK: child_drive=1

hw.module private @child(in %in : i1, out out : i1) {
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %sig = llhd.sig %false : i1
  llhd.drv %sig, %in after %eps : i1
  %val = llhd.prb %sig : i1
  hw.output %val : i1
}

hw.module @test() {
  %c1_i64 = hw.constant 1000000 : i64
  %c3_i64 = hw.constant 3000000 : i64
  %true = hw.constant true
  %false = hw.constant false
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_pre = sim.fmt.literal "child_drive="
  %fmt_nl = sim.fmt.literal "\0A"

  %in_sig = llhd.sig %false : i1
  %in_val = llhd.prb %in_sig : i1
  %inst.out = hw.instance "u" @child(in: %in_val : i1) -> (out: i1)

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.drv %in_sig, %true after %eps : i1
    llhd.halt
  }

  llhd.process {
    llhd.wait (%inst.out : i1), ^bb1
  ^bb1:
    %fmt_val = sim.fmt.bin %inst.out : i1
    %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }

  llhd.process {
    %delay = llhd.int_to_time %c3_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    sim.terminate failure, quiet
    llhd.halt
  }

  hw.output
}
