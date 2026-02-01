// RUN: circt-sim %s | FileCheck %s

// CHECK: pre=0
// CHECK: post=1

hw.module @test() {
  %c1_i64 = hw.constant 1000000 : i64
  %c2_i64 = hw.constant 2000000 : i64
  %c4_i64 = hw.constant 4000000 : i64
  %true = hw.constant true
  %false = hw.constant false
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_pre = sim.fmt.literal "pre="
  %fmt_post = sim.fmt.literal "post="
  %fmt_nl = sim.fmt.literal "\0A"

  %in_sig = llhd.sig %false : i1
  %en_sig = llhd.sig %false : i1
  %out_sig = llhd.sig %false : i1

  %in_val = llhd.prb %in_sig : i1
  %en_val = llhd.prb %en_sig : i1
  llhd.drv %out_sig, %in_val after %eps if %en_val : i1

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.drv %in_sig, %true after %eps : i1
    llhd.halt
  }

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %val = llhd.prb %out_sig : i1
    %fmt_val = sim.fmt.bin %val : i1
    %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    llhd.halt
  }

  llhd.process {
    %delay = llhd.int_to_time %c2_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.drv %en_sig, %true after %eps : i1
    llhd.halt
  }

  llhd.process {
    %obs = llhd.prb %out_sig : i1
    llhd.wait (%obs : i1), ^bb1
  ^bb1:
    %val = llhd.prb %out_sig : i1
    %fmt_val = sim.fmt.bin %val : i1
    %fmt_out = sim.fmt.concat (%fmt_post, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }

  llhd.process {
    %delay = llhd.int_to_time %c4_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    sim.terminate failure, quiet
    llhd.halt
  }

  hw.output
}
