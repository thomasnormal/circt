// RUN: circt-sim --max-process-steps=50 %s | FileCheck %s

// CHECK: loop-ok

hw.module @test() {
  %c3 = hw.constant 3 : i2
  %c1 = hw.constant 1 : i2
  %c0 = hw.constant 0 : i2
  %fmt = sim.fmt.literal "loop-ok\0A"

  llhd.process {
    cf.br ^bb1(%c3 : i2)
  ^bb1(%i: i2):
    %done = comb.icmp eq %i, %c0 : i2
    cf.cond_br %done, ^bb2, ^bb3
  ^bb3:
    %next = comb.sub %i, %c1 : i2
    cf.br ^bb1(%next : i2)
  ^bb2:
    sim.proc.print %fmt
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
