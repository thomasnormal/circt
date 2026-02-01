// RUN: circt-sim %s | FileCheck %s

// CHECK: sig_out=1

hw.module @test() {
  %false = hw.constant false
  %true = hw.constant true
  %c1_i64 = hw.constant 1000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_pre = sim.fmt.literal "sig_out="
  %fmt_nl = sim.fmt.literal "\0A"

  %sig_in = llhd.sig %false : i1
  %sig_out = llhd.sig %false : i1

  %proc_val = llhd.process -> i1 {
    llhd.wait yield (%true : i1), delay %eps, ^bb1
  ^bb1:
    llhd.halt %true : i1
  }

  %in_val = llhd.prb %sig_in : i1
  %derived = comb.and %proc_val, %in_val : i1
  llhd.drv %sig_out, %derived after %eps : i1

  llhd.process {
    %t1 = llhd.int_to_time %c1_i64
    llhd.wait delay %t1, ^bb1
  ^bb1:
    llhd.drv %sig_in, %true after %eps : i1
    llhd.wait delay %t1, ^bb2
  ^bb2:
    %out = llhd.prb %sig_out : i1
    %fmt_val = sim.fmt.dec %out : i1
    %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
