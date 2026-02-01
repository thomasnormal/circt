// RUN: circt-sim %s | FileCheck %s

// CHECK: drive_from_proc=1

hw.module @test() {
  %false = hw.constant false
  %true = hw.constant true
  %c1_i64 = hw.constant 1000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_pre = sim.fmt.literal "drive_from_proc="
  %fmt_nl = sim.fmt.literal "\0A"

  %sig = llhd.sig %false : i1

  %proc_val = llhd.process -> i1 {
    llhd.wait yield (%true : i1), delay %eps, ^bb1
  ^bb1:
    llhd.halt %true : i1
  }

  %derived = comb.xor %proc_val, %false : i1
  llhd.drv %sig, %derived after %eps : i1

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %val = llhd.prb %sig : i1
    %fmt_val = sim.fmt.dec %val : i1
    %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
