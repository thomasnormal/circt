// RUN: circt-sim %s | FileCheck %s

// CHECK: and=5
// CHECK: or=5
// CHECK: xor=5

hw.module @test() {
  %val = hw.constant 5 : i4
  %zero = hw.constant 0 : i4
  %ones = hw.constant 15 : i4

  %fmt_and = sim.fmt.literal "and="
  %fmt_or = sim.fmt.literal "or="
  %fmt_xor = sim.fmt.literal "xor="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %and = comb.and %val, %ones, %ones : i4
    %or = comb.or %val, %zero, %zero : i4
    %xor = comb.xor %val, %zero, %zero : i4

    %and_fmt = sim.fmt.dec %and : i4
    %or_fmt = sim.fmt.dec %or : i4
    %xor_fmt = sim.fmt.dec %xor : i4

    %and_out = sim.fmt.concat (%fmt_and, %and_fmt, %fmt_nl)
    %or_out = sim.fmt.concat (%fmt_or, %or_fmt, %fmt_nl)
    %xor_out = sim.fmt.concat (%fmt_xor, %xor_fmt, %fmt_nl)
    sim.proc.print %and_out
    sim.proc.print %or_out
    sim.proc.print %xor_out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
