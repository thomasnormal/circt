// RUN: circt-sim %s | FileCheck %s

// CHECK: slice=127

hw.module @test() {
  %val = hw.constant 0x00000000000000000000007F00000000 : i128
  %fmt_slice = sim.fmt.literal "slice="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %slice = comb.extract %val from 32 : (i128) -> i8
    %slice_fmt = sim.fmt.dec %slice : i8
    %out = sim.fmt.concat (%fmt_slice, %slice_fmt, %fmt_nl)
    sim.proc.print %out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
