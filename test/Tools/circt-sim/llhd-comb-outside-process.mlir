// RUN: circt-sim %s | FileCheck %s

// CHECK: not_a=1

hw.module @test() {
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %c1_i64 = hw.constant 1000000 : i64
  %true = hw.constant true
  %false = hw.constant false

  %sig = llhd.sig %false : i1
  %sig_val = llhd.prb %sig : i1
  %not_a = comb.xor %sig_val, %true : i1

  %fmt_pre = sim.fmt.literal "not_a="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    llhd.drv %sig, %false after %eps : i1
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %fmt_val = sim.fmt.bin %not_a : i1
    %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
