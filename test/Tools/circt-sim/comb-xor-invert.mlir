// RUN: circt-sim %s | FileCheck %s

// CHECK: xor1=10
// CHECK: xor2=5

hw.module @test() {
  %val = hw.constant 5 : i4
  %ones = hw.constant 15 : i4
  %fmt_xor1 = sim.fmt.literal "xor1="
  %fmt_xor2 = sim.fmt.literal "xor2="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %xor1 = comb.xor %val, %ones : i4
    %xor2 = comb.xor %val, %ones, %ones : i4
    %xor1_fmt = sim.fmt.dec %xor1 : i4
    %xor2_fmt = sim.fmt.dec %xor2 : i4
    %out1 = sim.fmt.concat (%fmt_xor1, %xor1_fmt, %fmt_nl)
    %out2 = sim.fmt.concat (%fmt_xor2, %xor2_fmt, %fmt_nl)
    sim.proc.print %out1
    sim.proc.print %out2
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
